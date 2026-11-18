import type { Config } from "tailwindcss";
import animate from "tailwindcss-animate";

const config: Config = {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: { "2xl": "1600px" },
    },
    extend: {
      colors: {
        // ── Surfaces ──
        background: {
          DEFAULT: "#F7F5F2",
          warm: "#F7F3E8",
        },
        surface: {
          DEFAULT: "#FFFFFF",
          muted: "#F2EFE9",
          hover: "#FAF8F4",
        },
        // ── Text ──
        ink: {
          DEFAULT: "#172B50",
          secondary: "#596579",
          muted: "#7C8492",
          faint: "#A0A6B0",
        },
        // ── Primary (Burgundy) ──
        primary: {
          DEFAULT: "#8C263B",
          hover: "#762033",
          light: "#A8334A",
          faint: "#F5E8EB",
        },
        // ── Semantic ──
        border: {
          DEFAULT: "#E5DED2",
          strong: "#D4CBBB",
        },
        success: {
          DEFAULT: "#2F7668",
          light: "#E8F2F0",
        },
        warning: {
          DEFAULT: "#A36B27",
          light: "#F5EDE0",
        },
        error: {
          DEFAULT: "#B44747",
          light: "#F5E8E8",
        },
        // ── shadcn/ui token mapping ──
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        ring: "hsl(var(--ring))",
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        input: "hsl(var(--input))",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        heading: ["Inter", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      boxShadow: {
        subtle: "0 1px 3px 0 rgba(23, 43, 80, 0.06), 0 1px 2px 0 rgba(23, 43, 80, 0.04)",
        card: "0 2px 8px 0 rgba(23, 43, 80, 0.06), 0 0 1px 0 rgba(23, 43, 80, 0.12)",
        elevated: "0 8px 24px 0 rgba(23, 43, 80, 0.08), 0 0 1px 0 rgba(23, 43, 80, 0.12)",
        dropdown: "0 4px 16px 0 rgba(23, 43, 80, 0.10), 0 0 1px 0 rgba(23, 43, 80, 0.12)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "fade-in": {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        "slide-up": {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "slide-in-right": {
          from: { opacity: "0", transform: "translateX(12px)" },
          to: { opacity: "1", transform: "translateX(0)" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "fade-in": "fade-in 0.2s ease-out",
        "slide-up": "slide-up 0.25s cubic-bezier(0.16, 1, 0.3, 1)",
        "slide-in-right": "slide-in-right 0.2s cubic-bezier(0.16, 1, 0.3, 1)",
      },
    },
  },
  plugins: [animate],
};

export default config;
