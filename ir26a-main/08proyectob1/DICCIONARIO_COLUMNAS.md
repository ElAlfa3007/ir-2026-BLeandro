# Diccionario de Columnas
## Columnas Principales (Con Datos)

| Campo | Tipo | Descripción | Completitud |
|-------|------|-------------|-------------|
| `job_id` | String | Hash único MD5/SHA256 para deduplicación | 100.0% |
| `source` | String | Plataforma origen (coresignal, jooble, rapidapi1, rapidapi2) | 100.0% |
| `job_title` | String | Título del puesto de trabajo | 100.0% |
| `company` | String | Nombre de la empresa | 97.8% |
| `location` | String | Ubicación geográfica original | 99.8% |
| `location_final` | String | Ubicación geográfica normalizada (país) | 99.1% |
| `description` | Text | Descripción completa del puesto original | 89.1% |
| `skills` | Array | Habilidades técnicas y blandas extraídas originalmente | 4.0% |
| `careers_required` | String | Carrera universitaria requerida | 99.9% |
| `date_posted` | Date | Fecha de publicación original | 98.5% |
| `date_posted_norm` | Date | Fecha de publicación normalizada | 98.5% |
| `url` | String | Enlace a la oferta original | 99.9% |
| `career_tag` | String | Etiqueta de categoría de carrera procesada | 10.9% |
| `soft_skills_detected` | Array | Habilidades blandas detectadas automáticamente | 99.9% |
| `extraction_date` | Date | Fecha de extracción del sistema | 99.9% |
| `description_final` | Text | Descripción traducida,procesada y limpia del puesto | 89.1% |
| `EURACE_skills` | Array | Habilidades clasificadas según estándar EURACE | 29.8% |
| `initial_skills` | Array | Habilidades iniciales extraídas en procesamiento | 29.8% |

## Notas Importantes

### Columnas Vacías
- **Columnas**: `Unnamed: 17` hasta `Unnamed: 55` (39 columnas)
- **Estado**: Completamente vacías (0% datos)
- **Recomendación**: Pueden eliminarse sin pérdida de información

## Diccionario Python Simple

```python
dir_columnas = {
    "job_id": "Hash único MD5/SHA256 para deduplicación",
    "source": "Plataforma origen (coresignal, jooble, rapidapi1, rapidapi2)",
    "job_title": "Título del puesto de trabajo",
    "company": "Nombre de la empresa",
    "location": "Ubicación geográfica original",
    "location_final": "Ubicación geográfica normalizada (país)",
    "description": "Descripción completa del puesto original",
    "skills": "Habilidades técnicas y blandas extraídas originalmente",
    "careers_required": "Carrera universitaria requerida",
    "date_posted": "Fecha de publicación original",
    "date_posted_norm": "Fecha de publicación normalizada",
    "url": "Enlace a la oferta original",
    "career_tag": "Etiqueta de categoría de carrera procesada",
    "soft_skills_detected": "Habilidades blandas detectadas automáticamente",
    "extraction_date": "Fecha de extracción del sistema",
    "description_final": "Descripción traducida,procesada y limpia del puesto",
    "EURACE_skills": "Habilidades clasificadas según estándar EURACE",
    "initial_skills": "Habilidades iniciales extraídas en procesamiento",
}



