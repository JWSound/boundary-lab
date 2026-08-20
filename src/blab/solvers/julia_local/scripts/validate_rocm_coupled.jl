#!/usr/bin/env julia

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled

using LinearAlgebra, Printf, SparseArrays, StaticArrays

const FIXTURE_ROOT = normpath(
    joinpath(@__DIR__, "..", "..", "..", "..", "..", "tests", "fixtures"),
)

function relative_error(reference, candidate)
    scale = norm(reference)
    return norm(candidate - reference) / max(scale, eps(Float64))
end

function timed_build(f)
    value = nothing
    elapsed = @elapsed value = f()
    return value, elapsed
end

function timed_solve(system, radiator_tag)
    values = nothing
    elapsed = @elapsed values = solve_coupled_systems(
        system,
        [radiator_tag, radiator_tag];
        radiator_velocities=ComplexF32[1, 0.5],
    )
    return values, elapsed
end

function main()
    amdgpu = BeatEngineCore.amdgpu_module()
    amdgpu.functional() || error("AMDGPU.functional() is false.")
    amdgpu.functional(:rocblas) || error("rocBLAS is not functional.")
    amdgpu.functional(:rocsolver) || error("rocSOLVER is not functional.")

    quadrature_order = parse(Int, get(ENV, "BLAB_COUPLED_QUADRATURE_ORDER", "1"))
    singular_order = parse(Int, get(ENV, "BLAB_COUPLED_SINGULAR_ORDER", "1"))
    fem_mesh = load_gmsh41_volume(joinpath(FIXTURE_ROOT, "femvolume.msh"), 0.001f0)
    bem_mesh = load_gmsh22_with_tags(
        joinpath(FIXTURE_ROOT, "exterior_conforming.msh"),
        0.001f0,
    )
    interface_map = build_conforming_interface_map(
        fem_mesh,
        bem_mesh,
        physical_tag(fem_mesh, 2, "Interface"),
        2,
    )
    radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
    transducer = ElectrodynamicTransducer{Float32}(
        "component:rocm-validation",
        [radiator_tag],
        Float32[1],
        [1],
        Float32[-1],
        SVector(0f0, 0f0, 1f0),
        2f0,
        1,
        6f0,
        0.0005f0,
        7f0,
        0.015f0,
        0.0005f0,
        1f0,
    )
    common_options = (
        quadrature_order=quadrature_order,
        singular_order=singular_order,
        bulk_loss_factor=0.01f0,
        transducers=[transducer],
    )

    cpu_system = diagnostic_system = rocm_system = condensed_system = nothing
    try
        cpu_system, cpu_build_s = timed_build() do
            build_coupled_system(
                fem_mesh,
                bem_mesh,
                interface_map,
                500f0,
                343f0,
                1.21f0;
                common_options...,
                bem_backend=:cpu,
                validation_diagnostics=false,
                static_condensation=false,
            )
        end
        diagnostic_system, diagnostic_build_s = timed_build() do
            build_coupled_system(
                fem_mesh,
                bem_mesh,
                interface_map,
                500f0,
                343f0,
                1.21f0;
                common_options...,
                bem_backend=:rocm,
                validation_diagnostics=true,
                static_condensation=false,
            )
        end
        rocm_system, rocm_build_s = timed_build() do
            build_coupled_system(
                fem_mesh,
                bem_mesh,
                interface_map,
                500f0,
                343f0,
                1.21f0;
                common_options...,
                bem_backend=:rocm,
                validation_diagnostics=false,
                static_condensation=false,
            )
        end
        condensed_system, condensed_build_s = timed_build() do
            build_coupled_system(
                fem_mesh,
                bem_mesh,
                interface_map,
                500f0,
                343f0,
                1.21f0;
                common_options...,
                bem_backend=:rocm,
                validation_diagnostics=false,
                static_condensation=true,
            )
        end

        cpu_solutions, cpu_solve_s = timed_solve(cpu_system, radiator_tag)
        diagnostic_solutions, diagnostic_solve_s = timed_solve(diagnostic_system, radiator_tag)
        rocm_solutions, rocm_solve_s = timed_solve(rocm_system, radiator_tag)
        condensed_solutions, condensed_solve_s = timed_solve(condensed_system, radiator_tag)
        cpu_solution = cpu_solutions[1]
        diagnostic_solution = diagnostic_solutions[1]
        rocm_solution = rocm_solutions[1]
        condensed_solution = condensed_solutions[1]
        voltage_excitation = (
            kind=:voltage,
            fem_boundary_tags=Int[],
            fem_boundary_weights=Float32[],
            bem_source_index=0,
            transducer_index=1,
            amplitude=ComplexF32(2.83, 0),
        )
        cpu_voltage = only(solve_coupled_excitations(cpu_system, [voltage_excitation]))
        diagnostic_voltage = only(
            solve_coupled_excitations(diagnostic_system, [voltage_excitation]),
        )
        rocm_voltage = only(solve_coupled_excitations(rocm_system, [voltage_excitation]))
        condensed_voltage = only(
            solve_coupled_excitations(condensed_system, [voltage_excitation]),
        )

        checks = Dict(
            "diagnostic_fem" => relative_error(cpu_solution.fem_pressure, diagnostic_solution.fem_pressure),
            "diagnostic_bem" => relative_error(cpu_solution.bem_pressure, diagnostic_solution.bem_pressure),
            "diagnostic_flux" => relative_error(cpu_solution.interface_flux, diagnostic_solution.interface_flux),
            "rocm_fem" => relative_error(cpu_solution.fem_pressure, rocm_solution.fem_pressure),
            "rocm_bem" => relative_error(cpu_solution.bem_pressure, rocm_solution.bem_pressure),
            "rocm_flux" => relative_error(cpu_solution.interface_flux, rocm_solution.interface_flux),
            "rocm_multi_rhs" => relative_error(
                0.5f0 .* rocm_solution.bem_pressure,
                rocm_solutions[2].bem_pressure,
            ),
            "diagnostic_voltage_bem" => relative_error(
                cpu_voltage.bem_pressure,
                diagnostic_voltage.bem_pressure,
            ),
            "diagnostic_voltage_fem" => relative_error(
                cpu_voltage.fem_pressure,
                diagnostic_voltage.fem_pressure,
            ),
            "diagnostic_voltage_flux" => relative_error(
                cpu_voltage.interface_flux,
                diagnostic_voltage.interface_flux,
            ),
            "diagnostic_voltage_velocity" => relative_error(
                cpu_voltage.diaphragm_velocity,
                diagnostic_voltage.diaphragm_velocity,
            ),
            "diagnostic_voltage_current" => relative_error(
                cpu_voltage.voice_coil_current,
                diagnostic_voltage.voice_coil_current,
            ),
            "rocm_voltage_fem" => relative_error(
                cpu_voltage.fem_pressure,
                rocm_voltage.fem_pressure,
            ),
            "rocm_voltage_bem" => relative_error(
                cpu_voltage.bem_pressure,
                rocm_voltage.bem_pressure,
            ),
            "rocm_voltage_flux" => relative_error(
                cpu_voltage.interface_flux,
                rocm_voltage.interface_flux,
            ),
            "rocm_voltage_velocity" => relative_error(
                cpu_voltage.diaphragm_velocity,
                rocm_voltage.diaphragm_velocity,
            ),
            "rocm_voltage_current" => relative_error(
                cpu_voltage.voice_coil_current,
                rocm_voltage.voice_coil_current,
            ),
            "condensed_fem" => relative_error(
                rocm_solution.fem_pressure,
                condensed_solution.fem_pressure,
            ),
            "condensed_bem" => relative_error(
                rocm_solution.bem_pressure,
                condensed_solution.bem_pressure,
            ),
            "condensed_flux" => relative_error(
                rocm_solution.interface_flux,
                condensed_solution.interface_flux,
            ),
            "condensed_multi_rhs" => relative_error(
                0.5f0 .* condensed_solution.bem_pressure,
                condensed_solutions[2].bem_pressure,
            ),
            "condensed_voltage_fem" => relative_error(
                rocm_voltage.fem_pressure,
                condensed_voltage.fem_pressure,
            ),
            "condensed_voltage_bem" => relative_error(
                rocm_voltage.bem_pressure,
                condensed_voltage.bem_pressure,
            ),
            "condensed_voltage_flux" => relative_error(
                rocm_voltage.interface_flux,
                condensed_voltage.interface_flux,
            ),
            "condensed_voltage_velocity" => relative_error(
                rocm_voltage.diaphragm_velocity,
                condensed_voltage.diaphragm_velocity,
            ),
            "condensed_voltage_current" => relative_error(
                rocm_voltage.voice_coil_current,
                condensed_voltage.voice_coil_current,
            ),
        )

        points = SVector{3,Float32}[
            SVector(0.20f0, 0.13f0, 0.30f0),
            SVector(-0.27f0, 0.08f0, 0.24f0),
        ]
        cpu_field = evaluate_galerkin_field_cpu(
            points,
            bem_mesh,
            cpu_solution.bem_pressure,
            cpu_solution.bem_neumann,
            cpu_system.wavenumber,
            cpu_system.field_cache,
        )
        rocm_field = evaluate_galerkin_field_rocm(
            points,
            bem_mesh,
            rocm_solution.bem_pressure,
            rocm_solution.bem_neumann,
            rocm_system.wavenumber,
            rocm_system.field_cache,
        )
        checks["rocm_field"] = relative_error(cpu_field, rocm_field)
        condensed_field = evaluate_galerkin_field_rocm(
            points,
            bem_mesh,
            condensed_solution.bem_pressure,
            condensed_solution.bem_neumann,
            condensed_system.wavenumber,
            condensed_system.field_cache,
        )
        checks["condensed_field"] = relative_error(rocm_field, condensed_field)

        println("BEAT coupled ROCm validation at 500 Hz (q$quadrature_order/s$singular_order)")
        @printf "  %-12s build=%8.3f s solve=%8.3f s linear=%s\n" "CPU" cpu_build_s cpu_solve_s cpu_system.linear_backend
        @printf "  %-12s build=%8.3f s solve=%8.3f s linear=%s\n" "ROCm diag" diagnostic_build_s diagnostic_solve_s diagnostic_system.linear_backend
        @printf "  %-12s build=%8.3f s solve=%8.3f s linear=%s\n" "ROCm" rocm_build_s rocm_solve_s rocm_system.linear_backend
        @printf "  %-12s build=%8.3f s solve=%8.3f s linear=%s order=%d/%d\n" "ROCm Schur" condensed_build_s condensed_solve_s condensed_system.linear_backend condensed_system.solved_system_order condensed_system.full_system_order
        for name in sort!(collect(keys(checks)))
            @printf "  %-24s %.6e\n" name checks[name]
        end
        @printf "  pressure continuity     %.6e\n" rocm_solution.pressure_continuity_error
        @printf "  flux conservation       %.6e\n" rocm_solution.flux_conservation_error
        @printf "  CPU voltage flux norm   %.6e\n" norm(cpu_voltage.interface_flux)
        @printf "  ROCm voltage flux abs   %.6e\n" norm(rocm_voltage.interface_flux - cpu_voltage.interface_flux)
        @printf "  diag voltage flux abs   %.6e\n" norm(diagnostic_voltage.interface_flux - cpu_voltage.interface_flux)
        @printf "  diag voltage residual   %.6e\n" diagnostic_voltage.relative_residual

        maximum(checks[name] for name in (
            "diagnostic_fem",
            "diagnostic_bem",
            "diagnostic_flux",
            "diagnostic_voltage_bem",
            "diagnostic_voltage_fem",
            "diagnostic_voltage_velocity",
            "diagnostic_voltage_current",
        )) < 5e-4 ||
            error("ROCm diagnostic path exceeds the 5e-4 parity gate.")
        maximum(checks[name] for name in (
            "rocm_fem",
            "rocm_bem",
            "rocm_flux",
            "rocm_field",
            "rocm_voltage_fem",
            "rocm_voltage_bem",
            "rocm_voltage_velocity",
            "rocm_voltage_current",
        )) < 5e-4 ||
            error("ROCm monolithic path exceeds the 5e-4 parity gate.")
        # This intentionally simple voltage fixture is very poorly conditioned
        # in FP32 (cond(A, 1) is about 3.8e9).  Even the ROCm-assembled/CPU-LU
        # diagnostic path amplifies ~1e-6 BEM matrix differences into ~1.4e-3
        # interface-flux variation, while the physical pressure, velocity, and
        # current outputs remain inside the strict gate above.
        max(
            checks["diagnostic_voltage_flux"],
            checks["rocm_voltage_flux"],
        ) < 2e-3 || error("Ill-conditioned voltage interface-flux parity exceeds 2e-3.")
        diagnostic_voltage.relative_residual < 2e-6 ||
            error("Diagnostic voltage residual exceeds 2e-6.")
        checks["rocm_multi_rhs"] < 1e-5 || error("ROCm multi-RHS linearity exceeds 1e-5.")
        condensed_system.formulation == :fem_interface_condensed ||
            error("ROCm hybrid solve did not select the condensed formulation.")
        condensed_system.solved_system_order < condensed_system.full_system_order ||
            error("ROCm hybrid solve did not reduce the system order.")
        maximum(checks[name] for name in (
            "condensed_fem",
            "condensed_bem",
            "condensed_flux",
            "condensed_field",
            "condensed_voltage_fem",
            "condensed_voltage_bem",
            "condensed_voltage_velocity",
            "condensed_voltage_current",
        )) < 1e-3 || error("ROCm hybrid condensation exceeds the 1e-3 parity gate.")
        checks["condensed_voltage_flux"] < 2e-3 ||
            error("ROCm condensed voltage flux exceeds the 2e-3 conditioning gate.")
        checks["condensed_multi_rhs"] < 1e-5 ||
            error("ROCm condensed multi-RHS linearity exceeds 1e-5.")
        rocm_solution.pressure_continuity_error < 1e-4 || error("Pressure continuity exceeds 1e-4.")
        rocm_solution.flux_conservation_error < 1e-4 || error("Flux conservation exceeds 1e-4.")
    finally
        cpu_system === nothing || release_coupled_system!(cpu_system)
        diagnostic_system === nothing || release_coupled_system!(diagnostic_system)
        rocm_system === nothing || release_coupled_system!(rocm_system)
        condensed_system === nothing || release_coupled_system!(condensed_system)
    end
end

main()
