CREATE TABLE IF NOT EXISTS keywords (
    id SERIAL PRIMARY KEY,
    keyword TEXT NOT NULL,
    severity INTEGER NOT NULL DEFAULT 1,
    is_regex BOOLEAN DEFAULT FALSE
);

-- Sample inserts
INSERT INTO keywords (keyword, severity, is_regex) VALUES
('idiot', 3, FALSE),
('dumb', 2, FALSE),
('hate', 2, FALSE),
('f.u.c.k', 5, TRUE);
