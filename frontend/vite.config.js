import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to FastAPI so you don't need CORS in dev
      "/sentiment": "http://127.0.0.1:8000",
      "/agent": "http://127.0.0.1:8000",
    },
  },
});
