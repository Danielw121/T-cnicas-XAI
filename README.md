# Taller XAI – Explicabilidad en Modelos de Crédito Bancario

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Open%20in-Colab-yellow.svg)](https://colab.research.google.com/)
[![Licencia](https://img.shields.io/badge/Licencia-MIT-green.svg)](LICENSE)

---

## Descripción

Este repositorio contiene el desarrollo completo del **Taller de Explicabilidad (XAI)** aplicado a un sistema de evaluación de créditos bancarios. El objetivo es demostrar cómo las técnicas de IA Explicable permiten comprender, auditar y comunicar las decisiones de modelos de Machine Learning, con especial atención a los riesgos éticos y sociales de sistemas automatizados de decisión crediticia.

---

## Estructura del Repositorio

```
├── XAI_Creditos_Bancarios.ipynb   # Notebook principal (Google Colab)
├── Creditos_Bancarios.csv          # Dataset de evaluación crediticia
├── README.md                       # Este archivo
└── Técnicas_XAI.docx              # Documento de instrucciones del taller
```

---

## 🎯 Objetivos del Taller

- Implementar modelos de ML aplicados a un problema real considerando calidad de datos y mitigación de sesgos.
- Aplicar técnicas de **explicabilidad (XAI)** para mejorar la transparencia del modelo.
- Reflexionar sobre los principios éticos en el diseño de sistemas automatizados.
- Documentar el flujo completo del proyecto con evaluación y comunicación de resultados.

---

## 📊 Dataset: `Creditos_Bancarios.csv`

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `edad` | Numérica | Edad del solicitante (18–64 años) |
| `ingresos` | Numérica | Ingresos mensuales en USD |
| `educacion` | Categórica | Nivel educativo (Primaria, Secundaria, Universitaria, Postgrado) |
| `zona` | Categórica | Zona geográfica (Urbana, Periferia, Rural) |
| `historial_crediticio` | Categórica | Historial previo (Bueno, Regular, Malo) |
| `estado_civil` | Categórica | Estado civil (Casado, Soltero, Divorciado) |
| `sexo` | Categórica | Sexo del solicitante (Masculino, Femenino) |
| `ocupacion` | Categórica | Tipo de empleo (Empleado, Independiente, Desempleado) |
| `resultado_credito_anterior` | Binaria | **Target**: 1 = Aprobado, 0 = Rechazado |

- **1500 registros** · **Sin valores nulos** · **64.4% aprobaciones**

---

## 🤖 Modelos Entrenados

| Modelo | Descripción |
|--------|-------------|
| **Random Forest** | 200 árboles, max_depth=8. Modelo principal para XAI. |
| **Regresión Logística** | Modelo de referencia interpretable por coeficientes. |

---

## 🔍 Técnicas XAI Aplicadas

### 1. 📊 Permutation Feature Importance
Mide la reducción en el rendimiento del modelo al permutar aleatoriamente cada variable. Más robusto que la importancia Gini por estar libre de sesgo hacia variables con muchas categorías.

### 2. 🎯 SHAP (Shapley Values)
Basado en teoría de juegos cooperativos. Asigna a cada variable una contribución justa a la predicción. Se aplican:
- **Summary Plot (Beeswarm)**: distribución del impacto de cada variable.
- **Bar Plot**: importancia global promedio.
- **Dependence Plot**: efecto de una variable específica.
- **Waterfall Plot**: explicación de predicciones individuales.

### 3. 🔬 LIME (Local Interpretable Model-Agnostic Explanations)
Ajusta un modelo lineal simple en la vecindad de cada predicción individual. Permite explicar **por qué** el modelo tomó una decisión concreta para un cliente específico.

### 4. 📈 Partial Dependence Plots (PDP + ICE)
Muestra el efecto marginal de cada variable sobre la probabilidad predicha, manteniendo las demás constantes. La curva ICE muestra variabilidad entre individuos.

---

## 🚀 Cómo Ejecutar

### Opción 1: Google Colab (Recomendado)

1. Subir `Creditos_Bancarios.csv` a tu Google Drive o directamente a Colab.
2. Abrir `XAI_Creditos_Bancarios.ipynb` en Google Colab.
3. Ejecutar la celda de instalación de dependencias (`!pip install shap lime ...`).
4. Ejecutar todas las celdas en orden.

### Opción 2: Entorno Local

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>

# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Instalar dependencias
pip install jupyter shap lime scikit-learn pandas numpy matplotlib seaborn

# Lanzar Jupyter
jupyter notebook XAI_Creditos_Bancarios.ipynb
```

---

## 📋 Contenido del Notebook

| Sección | Descripción |
|---------|-------------|
| 1. Instalación | Dependencias necesarias |
| 2. EDA | Exploración del dataset y distribuciones |
| 3. Sesgos | Análisis de fairness por sexo, zona y educación |
| 4. Preprocesamiento | Encoding, scaling, train/test split |
| 5. Entrenamiento | Random Forest + Regresión Logística |
| 6. Evaluación | Métricas, curvas ROC, matrices de confusión |
| 7.1 PFI | Permutation Feature Importance |
| 7.2 SHAP | Valores SHAP globales y locales |
| 7.3 LIME | Explicaciones locales por instancia |
| 7.4 PDP | Partial Dependence Plots + ICE |
| 8. Individual | Explicaciones para 2 clientes específicos |
| 9. Comparación | Ranking de variables entre técnicas |
| 10. Ética | Análisis de riesgos y recomendaciones |
| 11. Reflexión | Conclusiones del taller |

---

## 🏆 Hallazgos Principales

### Variables más importantes (consenso entre técnicas)
1. 🥇 `historial_crediticio` — el predictor dominante en todos los métodos
2. 🥈 `ingresos` — segunda variable en importancia
3. 🥉 `edad` — con efectos no lineales identificados por PDP

### Riesgos Éticos Identificados
- **Perpetuación de sesgos**: datos históricos reflejan decisiones previas potencialmente discriminatorias.
- **Proxy discrimination**: variables como `zona` y `ocupacion` pueden correlacionar con características protegidas.
- **Exclusión sistemática**: personas sin historial crediticio (jóvenes, migrantes) quedan penalizadas estructuralmente.
- **Opacidad sin XAI**: sin explicabilidad, los solicitantes rechazados no tienen forma de conocer ni cuestionar los motivos.

## 📚 Referencias

- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. NIPS.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why should I trust you?": Explaining the predictions of any classifier*. KDD.
- Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.). [christophm.github.io/interpretable-ml-book](https://christophm.github.io/interpretable-ml-book/)
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5–32.

---

## 👥 Autores

> Daniel Wong
> Luigy David Miranda Sandoval
> Jhustin Orozco Rocha
---

## 📄 Licencia

Este proyecto está bajo la licencia [MIT](LICENSE).
