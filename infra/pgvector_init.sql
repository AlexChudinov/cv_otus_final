CREATE EXTENSION vector;

CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    split TEXT NOT NULL,
    filename TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS image_embeddings (
    image_id INT NOT NULL,
    embedding VECTOR(768) NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS image_embeddings_image_id_idx ON image_embeddings(image_id);
CREATE UNIQUE INDEX IF NOT EXISTS images_filename_idx ON images(filename);
