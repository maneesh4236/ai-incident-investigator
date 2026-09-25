import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        base: "#0A0E13",
        panel: "#101720",
        panel2: "#141D28",
        line: "#1E2A36",
        ink: "#E8EDF2",
        muted: "#8B98A8",
        faint: "#5A6675",
        signal: {
          DEFAULT: "#F0A639",
          dim: "#7A5A28",
        },
        trace: {
          DEFAULT: "#31C6AD",
          dim: "#1E4A42",
        },
        critical: {
          DEFAULT: "#E5555C",
          dim: "#5A2226",
        },
      },
      fontFamily: {
        sans: ["var(--font-plex-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-plex-mono)", "ui-monospace", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.02) inset",
      },
    },
  },
  plugins: [],
};
export default config;
