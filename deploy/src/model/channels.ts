import type { DeployChannel, EqualizerConfiguration, SourceConfiguration } from "./types";

export const DEFAULT_CHANNEL_ID = "channel-main";
export const EMPTY_EQUALIZER: EqualizerConfiguration = { filters: [] };
export const CHANNEL_COLORS = ["#9eb9ef", "#efad72", "#8fd19e", "#d39be8", "#e4d071", "#78c9d4"];

export function createDefaultChannel(): DeployChannel {
  return {
    id: DEFAULT_CHANNEL_ID,
    name: "Main",
    color: CHANNEL_COLORS[0],
    levelDb: 0,
    delayMs: 0,
    polarity: 1,
    muted: false,
    equalizer: { filters: [] },
  };
}

export function applyChannelProcessing(
  sources: readonly SourceConfiguration[],
  channels: readonly DeployChannel[],
): SourceConfiguration[] {
  const channelById = new Map(channels.map((channel) => [channel.id, channel]));
  const fallback = channels[0] ?? createDefaultChannel();
  return sources.map((source) => {
    const channel = channelById.get(source.channelId) ?? fallback;
    return {
      ...source,
      levelDb: source.levelDb + channel.levelDb,
      delayMs: source.delayMs + channel.delayMs,
      polarity: (source.polarity * channel.polarity) as 1 | -1,
      muted: channel.muted,
    };
  });
}
