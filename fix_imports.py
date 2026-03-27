import glob

files = glob.glob('gomoku_9x9/**/*.py', recursive=True)
replacements = [
    ('from gomoku.', 'from core.'),
    ('import gomoku.', 'import core.'),
    ('gomoku_weights.pth', 'model_weights.pth'),
    ('gomoku_training_log.csv', 'training_log.csv'),
    ('gomoku_training_curves.png', 'training_curves.png'),
]
for f in files:
    with open(f, 'r', encoding='utf-8') as fh:
        content = fh.read()
    orig = content
    for old, new in replacements:
        content = content.replace(old, new)
    if content != orig:
        with open(f, 'w', encoding='utf-8') as fh:
            fh.write(content)
        print(f'Fixed: {f}')
print('Done.')
