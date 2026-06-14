"""Glosario EN→ES de títulos y términos técnicos completos.

Se aplica por match exacto (case-insensitive, espacios colapsados) sobre el
texto COMPLETO de un bloque corto o de un título TOC.  Bypassea el modelo en
títulos que OPUS-MT traduce mal de forma sistemática ("Fault Tolerance" →
"Tolerance", "Data Warehousing" → "Data de datos Almacenamiento") y resuelve
los títulos de 1 palabra que hoy quedan en inglés porque el modelo loopea
con inputs tan cortos.

Para añadir términos: clave en minúsculas con espacios simples, valor con la
capitalización deseada.  Las claves nunca colisionan con texto de párrafo
porque el lookup requiere que TODO el bloque coincida.
"""
import re

_WS_RE = re.compile(r"\s+")

# Comillas/apóstrofes tipográficos → ASCII para normalizar claves de lookup
_QUOTE_MAP = str.maketrans({
    "’": "'", "‘": "'",   # ’ ‘
    "“": '"', "”": '"',   # “ ”
})

GLOSSARY: dict[str, str] = {
    # --- Secciones genéricas de cualquier libro técnico ---
    "preface": "Prefacio",
    "foreword": "Prólogo",
    "introduction": "Introducción",
    "conclusion": "Conclusión",
    "summary": "Resumen",
    "abstract": "Resumen",
    "index": "Índice",
    "glossary": "Glosario",
    "references": "Referencias",
    "bibliography": "Bibliografía",
    "appendix": "Apéndice",
    "acknowledgments": "Agradecimientos",
    "acknowledgements": "Agradecimientos",
    "table of contents": "Tabla de contenidos",
    "contents": "Contenido",
    "about the author": "Sobre el autor",
    "about the authors": "Sobre los autores",
    "further reading": "Lecturas adicionales",
    "related work": "Trabajo relacionado",
    "future work": "Trabajo futuro",
    "overview": "Visión general",
    "background": "Antecedentes",
    "motivation": "Motivación",
    "examples": "Ejemplos",
    "exercises": "Ejercicios",
    "notes": "Notas",
    "epilogue": "Epílogo",
    "prologue": "Prólogo",
    "interlude": "Interludio",
    "dedication": "Dedicatoria",
    "afterword": "Epílogo del autor",
    "author's note": "Nota del autor",
    # Palabras estructurales estilizadas que aparecen como bloques sueltos en
    # portadillas de novelas ("PART" y "ONE" son bloques separados)
    "part": "Parte",
    "chapter": "Capítulo",
    "one": "Uno",
    "two": "Dos",
    "three": "Tres",
    "four": "Cuatro",
    "five": "Cinco",
    "six": "Seis",
    "seven": "Siete",
    "eight": "Ocho",
    "nine": "Nueve",
    "ten": "Diez",
    "part one": "Primera parte",
    "part two": "Segunda parte",
    "part three": "Tercera parte",

    # --- Palabras-título de 1 palabra que OPUS-MT destroza (gerundios, etc.) ---
    # ("Writing" -> "Ww w", "Coding" -> "", "Roleplaying" -> "Rede de")
    "writing": "Escritura",
    "coding": "Programación",
    "reading": "Lectura",
    "tutoring": "Tutoría",
    "roleplay": "Juego de roles",
    "roleplaying": "Juego de roles",
    "education": "Educación",
    "maintenance": "Mantenimiento",
    "planning": "Planificación",
    "training": "Entrenamiento",
    "inference": "Inferencia",
    "evaluation": "Evaluación",
    "benchmarking": "Evaluación comparativa",
    "prompting": "Prompting",
    "sampling": "Muestreo",
    "embedding": "Embedding",
    "embeddings": "Embeddings",
    "tokenization": "Tokenización",
    "quantization": "Cuantización",
    "summarization": "Resumen",
    "translation": "Traducción",
    "classification": "Clasificación",
    "clustering": "Agrupamiento",
    "deployment": "Despliegue",
    "monitoring": "Monitoreo",
    "logging": "Registro",
    "caching": "Almacenamiento en caché",
    "routing": "Enrutamiento",
    "scaling": "Escalado",
    "debugging": "Depuración",
    "testing": "Pruebas",
    "optimization": "Optimización",
    "personalization": "Personalización",
    "memory": "Memoria",
    "guardrails": "Barreras de seguridad",
    "agents": "Agentes",
    "tools": "Herramientas",
    "retrieval": "Recuperación",
    "ranking": "Clasificación por relevancia",
    "indexing": "Indexación",
    "moderation": "Moderación",

    # --- Frases-sección frecuentes en libros de IA (AI Engineering) ---
    "image and video production": "Producción de imagen y vídeo",
    "conversational bots": "Bots conversacionales",
    "information aggregation": "Agregación de información",
    "data organization": "Organización de datos",
    "workflow automation": "Automatización de flujos de trabajo",
    "prompt engineering": "Ingeniería de prompts",
    "fine-tuning": "Ajuste fino",
    "foundation models": "Modelos fundacionales",
    "foundation model": "Modelo fundacional",
    "large language models": "Modelos de lenguaje grandes",
    "large language model": "Modelo de lenguaje grande",
    "ai engineering": "Ingeniería de IA",
    "exact evaluation": "Evaluación exacta",
    "dataset engineering": "Ingeniería de conjuntos de datos",
    "model selection": "Selección de modelos",
    "use cases": "Casos de uso",
    "case study": "Caso de estudio",
    "memex": "Memex",

    # --- Términos de sistemas/datos de 1-2 palabras (loops del modelo) ---
    "scalability": "Escalabilidad",
    "maintainability": "Mantenibilidad",
    "reliability": "Fiabilidad",
    "availability": "Disponibilidad",
    "consistency": "Consistencia",
    "durability": "Durabilidad",
    "fault tolerance": "Tolerancia a fallos",
    "replication": "Replicación",
    "partitioning": "Particionamiento",
    "transactions": "Transacciones",
    "storage": "Almacenamiento",
    "data warehousing": "Almacén de datos",
    "batch processing": "Procesamiento por lotes",
    "stream processing": "Procesamiento de flujos",
    "distributed data": "Datos distribuidos",
    "derived data": "Datos derivados",
    "understanding load": "Comprensión de la carga",
    "describing performance": "Descripción del rendimiento",
    "enforcing constraints": "Aplicación de restricciones",
    "coordination services": "Servicios de coordinación",
    "column-oriented storage": "Almacenamiento orientado a columnas",
    "multi-leader replication": "Replicación multilíder",
    "multi-region operation": "Operación multirregión",
    "preventing lost updates": "Prevención de actualizaciones perdidas",
    "storage and retrieval": "Almacenamiento y recuperación",
    "consistency and consensus": "Consistencia y consenso",

    # --- Títulos compuestos que el modelo destroza de forma reproducible ---
    "trade-offs in data systems architecture":
        "Compensaciones en la arquitectura de sistemas de datos",
    "defining nonfunctional requirements":
        "Definición de requisitos no funcionales",
    "operational versus analytical systems":
        "Sistemas operacionales versus analíticos",
    "characterizing transaction processing and analytics":
        "Caracterización del procesamiento de transacciones y analítica",
    "systems of record and derived data":
        "Sistemas de registro y datos derivados",
    "cloud versus self-hosting": "Nube versus alojamiento propio",
    "pros and cons of cloud services":
        "Pros y contras de los servicios en la nube",
    "cloud native system architecture":
        "Arquitectura de sistemas nativos de la nube",
    "operations in the cloud era": "Operaciones en la era de la nube",
    "distributed versus single-node systems":
        "Sistemas distribuidos versus de un solo nodo",
    "problems with distributed systems":
        "Problemas de los sistemas distribuidos",
    "microservices and serverless": "Microservicios y serverless",
    "cloud computing versus supercomputing":
        "Computación en la nube versus supercomputación",
    "data systems, law, and society": "Sistemas de datos, ley y sociedad",
    "case study: social network home timelines":
        "Caso de estudio: cronologías de inicio de redes sociales",
    "representing users, posts, and follows":
        "Representación de usuarios, publicaciones y seguimientos",
    "materializing and updating timelines":
        "Materialización y actualización de cronologías",
    "latency and response time": "Latencia y tiempo de respuesta",
    "average, median, and percentiles": "Media, mediana y percentiles",
    "use of response time metrics": "Uso de métricas de tiempo de respuesta",
    "reliability and fault tolerance": "Fiabilidad y tolerancia a fallos",
    "hardware and software faults": "Fallos de hardware y software",
    "humans and reliability": "Humanos y fiabilidad",
    "principles for scalability": "Principios de escalabilidad",
    "operability: making life easy for operations":
        "Operabilidad: facilitar la vida a operaciones",
    "simplicity: managing complexity": "Simplicidad: gestión de la complejidad",
    "evolvability: making change easy": "Evolucionabilidad: facilitar el cambio",
    "shared-memory, shared-disk, and shared-nothing architectures":
        "Arquitecturas de memoria compartida, disco compartido y nada compartido",
    "relational versus document models":
        "Modelos relacionales versus de documentos",
    "many-to-one and many-to-many relationships":
        "Relaciones muchos-a-uno y muchos-a-muchos",
    "dataflow through databases": "Flujo de datos a través de bases de datos",
    "solutions for replication lag":
        "Soluciones para el retraso de replicación",
    "the majority rules": "La regla de la mayoría",
    "data as assets and power": "Los datos como activos y poder",
    "data models and query languages":
        "Modelos de datos y lenguajes de consulta",
    "the trouble with distributed systems":
        "Los problemas de los sistemas distribuidos",
    "the future of data systems": "El futuro de los sistemas de datos",
    "who should read this book?": "¿Quién debería leer este libro?",
    "what's new in the second edition?":
        "¿Qué hay de nuevo en la segunda edición?",

    # --- Títulos TOC que siguieron mal tras la primera tanda (diff fase1c) ---
    "skewed workloads and relieving hot spots":
        "Cargas de trabajo sesgadas y alivio de puntos calientes",
    "shuffling data": "Redistribución de datos",
    "joins and grouping": "Joins y agrupación",
    "transmitting event streams": "Transmisión de flujos de eventos",
    "stream joins": "Joins de flujos",
    "batch and stream processing": "Procesamiento por lotes y de flujos",
    "detecting concurrent writes": "Detección de escrituras concurrentes",
    "two-phase commit": "Commit de dos fases",
    "byzantine faults": "Fallos bizantinos",
    "synchronous versus asynchronous replication":
        "Replicación síncrona versus asíncrona",
    "single-leader versus leaderless replication performance":
        "Rendimiento de replicación con un solo líder versus sin líder",
    "synchronous versus asynchronous networks":
        "Redes síncronas versus asíncronas",
    "dataframes, matrices, and arrays": "DataFrames, matrices y arrays",
    "triple stores and sparql": "Almacenes de tripletas y SPARQL",
    "protocol buffers": "Protocol Buffers",
    "when to use which model": "Cuándo usar cada modelo",
    "exactly-once message processing revisited":
        "Procesamiento de mensajes exactly-once revisitado",
    "observing derived state": "Observación del estado derivado",
    "aiming for correctness": "Apuntando a la corrección",
    "stars and snowflakes: schemas for analytics":
        "Estrellas y copos de nieve: esquemas para analítica",
    "sharding and secondary indexes": "Sharding e índices secundarios",
    "distributed job orchestration": "Orquestación distribuida de trabajos",
    "implementing linearizable systems":
        "Implementación de sistemas linearizables",

    # --- Fase 2: hallazgos de la inspección visual del PDF fase 1 ---
    "2nd edition": "2ª edición",
    "second edition": "Segunda edición",
    "comparing b-trees and lsm-trees": "Comparación de B-Trees y LSM-Trees",
    "the object-relational mismatch": "El desajuste objeto-relacional",
    "multi-column indexes": "Índices multicolumna",
    "keeping everything in memory": "Mantener todo en memoria",
    "storage and indexing for oltp": "Almacenamiento e indexado para OLTP",
    "data storage for analytics": "Almacenamiento de datos para analítica",
    "event sourcing and cqrs": "Event sourcing y CQRS",
    "graphs as data models": "Grafos como modelos de datos",
    "graph queries in sql": "Consultas de grafos en SQL",
    "the cypher query language": "El lenguaje de consulta Cypher",
    "sharding of key-value data": "Sharding de datos clave-valor",
    "normalization, denormalization, and joins":
        "Normalización, desnormalización y joins",
}


def _norm_key(text: str) -> str:
    text = text.translate(_QUOTE_MAP)
    text = _WS_RE.sub(" ", text).strip()
    return text.lower().rstrip(" .:,;")


def lookup(text: str) -> str | None:
    """Traducción de glosario cuando TODO el texto coincide con una entrada,
    o None si no hay match."""
    return GLOSSARY.get(_norm_key(text))
