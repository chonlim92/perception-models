/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'kpi-pass': '#10b981',
        'kpi-warn': '#f59e0b',
        'kpi-fail': '#ef4444',
      },
    },
  },
  plugins: [],
};
