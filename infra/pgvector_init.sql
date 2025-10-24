CREATE EXTENSION vector;

CREATE TABLE IF NOT EXISTS images (
    url TEXT PRIMARY KEY,
    filename TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS image_embeddings (
    image_id INT NOT NULL,
    embedding VECTOR(768) NOT NULL
);

CREATE TABLE IF NOT EXISTS image_prompts (
    image_id INT NOT NULL,
    prompt TEXT NOT NULL
);
