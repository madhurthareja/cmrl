import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/chat": {
        target: "http://localhost:8001",
        changeOrigin: true,
      },
      "/vqa": {
        target: "http://localhost:8001",
        changeOrigin: true,
      },
      "/curriculum": {
        target: "http://localhost:8001",
        changeOrigin: true,
      },
      "/training": {
        target: "http://localhost:8001",
        changeOrigin: true,
      },
      "/health": {
        target: "http://localhost:8001",
        changeOrigin: true,
      },
    },
  },
});
