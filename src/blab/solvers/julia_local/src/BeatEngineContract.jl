module BeatEngineContract

using JSON

export validate_system_request, worker_ready, validate_worker_submission

# The same engine-owned schema is loaded by the standalone Python validator.
const SCHEMA = JSON.parsefile(joinpath(@__DIR__, "..", "..", "beat_contract", "system-v1.schema.json"))
const WORKER = JSON.parsefile(joinpath(@__DIR__, "..", "..", "beat_contract", "worker-v1.json"))

function worker_ready(backends)
    info = deepcopy(WORKER)
    info["backends"] = backends
    info["runtime"] = Dict("julia_version" => string(VERSION))
    return info
end

function validate_worker_submission(submission)
    submission isa AbstractDict || error("BEAT worker command must be an object.")
    version = get(submission, "protocol_version", nothing)
    version isa Integer && !(version isa Bool) && version == WORKER["protocol"]["version"] ||
        error("Incompatible BEAT worker protocol; client must select protocol version 1.")
    operation = get(submission, "operation", nothing)
    operation in WORKER["operations"] || error("Unsupported BEAT worker operation: $operation")
    request = get(submission, "request", nothing)
    request isa AbstractString && !isempty(request) || error("BEAT worker command requires a request filename.")
    field, contract = operation == "solve" ? ("result_schema_version", "system_result") : ("field_array_schema_version", "field_array")
    selected = get(submission, field, nothing)
    selected isa Integer && !(selected isa Bool) && selected in WORKER["contracts"][contract] ||
        error("Unsupported BEAT worker $field; select an advertised version.")
    return nothing
end

fail(path, message) = error("BEAT contract $path: $message")

function finite_json(value, path)
    if value === nothing || value isa AbstractString || value isa Bool || value isa Integer
        return
    elseif value isa Real && isfinite(value)
        return
    elseif value isa AbstractVector
        for (i, item) in enumerate(value)
            finite_json(item, "$path[$(i - 1)]")
        end
    elseif value isa AbstractDict && all(key isa AbstractString for key in keys(value))
        for (key, item) in value
            finite_json(item, "$path.$key")
        end
    else
        fail(path, "must contain finite JSON values")
    end
end

function matches(value, kind)
    kind == "object" && return value isa AbstractDict
    kind == "array" && return value isa AbstractVector
    kind == "string" && return value isa AbstractString
    kind == "null" && return value === nothing
    number = value isa Real && !(value isa Bool)
    kind == "number" && return number
    kind == "integer" && return number && isinteger(value)
    error("Unknown BEAT schema type: $kind")
end

function validate(value, schema, path)
    if haskey(schema, "\$ref")
        schema = SCHEMA["\$defs"][replace(schema["\$ref"], "#/\$defs/" => "")]
    end
    kinds = get(schema, "type", [])
    kinds = kinds isa AbstractString ? [kinds] : kinds
    isempty(kinds) || any(matches(value, kind) for kind in kinds) || fail(path, "unexpected JSON type")
    haskey(schema, "const") && value != schema["const"] && fail(path, "unsupported version")
    haskey(schema, "enum") && !(value in schema["enum"]) && fail(path, "unsupported enum value")
    if value isa AbstractDict
        for key in get(schema, "required", [])
            haskey(value, key) || fail("$path.$key", "required field is missing")
        end
        properties = get(schema, "properties", Dict())
        for (key, item) in value
            if haskey(properties, key)
                validate(item, properties[key], "$path.$key")
            elseif get(schema, "additionalProperties", true) === false
                fail("$path.$key", "unknown field; use metadata/options for extensions")
            end
        end
    elseif value isa AbstractVector
        get(schema, "minItems", 0) <= length(value) <= get(schema, "maxItems", Inf) || fail(path, "invalid array length")
        if haskey(schema, "items")
            for (i, item) in enumerate(value)
                validate(item, schema["items"], "$path[$(i - 1)]")
            end
        end
    elseif value isa AbstractString
        length(value) >= get(schema, "minLength", 0) || fail(path, "string is empty")
    elseif value isa Real && !(value isa Bool)
        haskey(schema, "minimum") && value < schema["minimum"] && fail(path, "below minimum")
        haskey(schema, "exclusiveMinimum") && value <= schema["exclusiveMinimum"] && fail(path, "must exceed minimum")
    end
end

function unique_ids(values, path)
    length(values) == length(Set(values)) || fail(path, "duplicate identifiers")
end

function references(values, available, path)
    unique_ids(values, path)
    for value in values
        haskey(available, value) || fail(path, "unknown reference $value")
    end
end

function graph(system)
    collections = Dict()
    for name in ("meshes", "regions", "boundaries", "interfaces", "components", "excitation_ports")
        unique_ids([item["id"] for item in system[name]], "compiled_system.$name")
        collections[name] = Dict(item["id"] => item for item in system[name])
    end
    meshes, regions, boundaries, components = (collections[key] for key in ("meshes", "regions", "boundaries", "components"))
    for region in values(regions)
        references(region["mesh_ids"], meshes, "region $(region["id"]).mesh_ids")
        for group in region["volume_groups"]
            group["mesh_id"] in region["mesh_ids"] && group["dimension"] == 3 || fail("volume_groups", "must reference a volume group on a region mesh")
        end
    end
    for boundary in values(boundaries)
        references([boundary["region_id"]], regions, "boundary $(boundary["id"]).region_id")
        group = boundary["group"]
        group["mesh_id"] in regions[boundary["region_id"]]["mesh_ids"] && group["dimension"] == 2 || fail("boundary.group", "must reference a surface group on a region mesh")
    end
    for component in values(components)
        references(component["boundary_ids"], boundaries, "component $(component["id"]).boundary_ids")
    end
    for port in values(collections["excitation_ports"])
        references([port["component_id"]], components, "port $(port["id"]).component_id")
    end
    for interface in values(collections["interfaces"])
        references([interface["bounded_boundary_id"], interface["unbounded_boundary_id"]], boundaries, "interface $(interface["id"])")
        topology = interface["topology"]
        length(topology["fem_vertex_indices"]) == length(topology["fem_to_bem_vertex_indices"]) || fail("topology", "vertex mapping lengths differ")
        length(Set(length(topology[key]) for key in ("fem_face_indices", "bem_face_indices", "normal_sign"))) == 1 || fail("topology", "face mapping lengths differ")
    end
end

function validate_system_request(request)
    finite_json(request, "request")
    validate(request, SCHEMA["\$defs"]["solve_request"], "request")
    graph(request["compiled_system"])
    references(request["excitation_port_ids"], Dict(port["id"] => port for port in request["compiled_system"]["excitation_ports"]), "request.excitation_port_ids")
    unique_ids([output["id"] for output in request["outputs"]], "request.outputs")
    return nothing
end

end # module
