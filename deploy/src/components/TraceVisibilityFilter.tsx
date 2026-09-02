export interface TraceVisibilityItem {
  id: string;
  name: string;
  color: string;
}

export function TraceVisibilityFilter({ items, hiddenIds, onToggle }: {
  items: TraceVisibilityItem[];
  hiddenIds: ReadonlySet<string>;
  onToggle: (id: string) => void;
}) {
  return (
    <aside className="trace-filter" aria-label="Plot line visibility">
      <div className="trace-filter-heading">Plot lines</div>
      <div className="trace-filter-list">
        {items.map((item) => (
          <label key={item.id} className={hiddenIds.has(item.id) ? "trace-filter-row hidden" : "trace-filter-row"}>
            <input type="checkbox" checked={!hiddenIds.has(item.id)} onChange={() => onToggle(item.id)} aria-label={`Show ${item.name}`} />
            <i style={{ background: item.color }} />
            <span title={item.name}>{item.name}</span>
          </label>
        ))}
      </div>
    </aside>
  );
}
