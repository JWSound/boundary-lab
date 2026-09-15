import type { Fidelity } from "../model/types";

export function FidelitySwitcher({
  value,
  onChange,
  packageLevel,
  boundaryAvailable,
  boundaryUnavailableReason,
  coupledAvailable,
  coupledUnavailableReason,
}: {
  value: Fidelity;
  onChange: (value: Fidelity) => void;
  packageLevel: number;
  boundaryAvailable: boolean;
  boundaryUnavailableReason?: string;
  coupledAvailable: boolean;
  coupledUnavailableReason?: string;
}) {
  const levels: Array<{ id: Fidelity; label: string; level: number }> = [
    { id: "pattern", label: "Pattern", level: 1 },
    { id: "boundary", label: "Boundary", level: 2 },
    { id: "coupled", label: "Coupled", level: 3 },
  ];
  return (
    <div className="fidelity-switcher">
      {levels.map((item) => {
        const available = item.level <= packageLevel;
        const interactive = item.id === "pattern" ||
          (item.id === "boundary" && available && boundaryAvailable) ||
          (item.id === "coupled" && available && coupledAvailable);
        return (
          <button
            key={item.id}
            className={`${value === item.id ? "active" : ""} ${!interactive ? "engine-required" : ""}`}
            onClick={() => interactive && onChange(item.id)}
            title={interactive
              ? (item.id === "boundary"
                  ? "Exterior BEM with fixed distributed sources"
                  : item.id === "coupled"
                    ? "Reduced-order coupled FEM–BEM interiors and transducers"
                    : "Live complex pattern field with infinite rigid ground at y=0")
              : item.id === "boundary" && boundaryUnavailableReason
                ? boundaryUnavailableReason
                : item.id === "coupled" && coupledUnavailableReason
                  ? coupledUnavailableReason
                : available ? "This fidelity is not connected yet" : "Package does not contain this fidelity"}
          >
            <span>{item.label}</span>
            {item.id !== "pattern" && interactive && <small>CUDA</small>}
            {!interactive && item.id !== "pattern" && <small>{available ? "ENGINE" : "N/A"}</small>}
          </button>
        );
      })}
    </div>
  );
}
