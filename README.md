# NeuroViz3D
Visualizador 3D de redes neuronales en OpenGL, con entrenamiento real en PyTorch.

## Fases
- [x] Fase 1: Ventana + triángulo (setup pipeline OpenGL)
- [x] Fase 2: Esferas + líneas (arquitectura fija)
- [x] Fase 3: Shaders custom (color/grosor por peso, iluminación Phong)
- [x] Fase 4: Cargar pesos reales exportados desde PyTorch (Iris)
- [x] Fase 5: Animación del forward pass
- [x] Fase 6: Loss/accuracy en tiempo real durante entrenamiento
- [x] Fase 7: Embeddings 3D (PCA/t-SNE) con transición animada
- [x] Fase 8: Pulido — cámara libre, picking de neuronas, UI con ImGui (completa)

## Controles
- Click + arrastrar: rota la cámara
- Scroll: zoom
- SPACE: reproduce el forward pass animado (modo red)
- T: reproduce el historial de entrenamiento (modo red)
- E: alterna entre modo red y modo embedding
- P: alterna entre proyección PCA y t-SNE (con transición animada, solo en modo embedding)
- C: alterna entre cámara orbital y cámara libre (WASD + mouse-look)
- WASD: mueve la cámara libre (solo en modo FREE)
- R / F: sube / baja la cámara libre (solo en modo FREE)
- Click derecho: selecciona una neurona (solo en modo NETWORK + cámara ORBITAL), resalta con glow y muestra capa/índice/activación en consola
- Panel ImGui (arriba a la izquierda): info general (FPS, sample activo, entrenamiento), datos de la neurona seleccionada, botones para alternar modo y camara, sliders de velocidad/sensibilidad de camara libre
- ESC: salir
## Demo
![demo](\neuro3D.png)