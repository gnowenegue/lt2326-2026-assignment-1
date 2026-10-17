import os
import shutil

import opencc

SETS = ['train', 'valid', 'test']
converter = opencc.OpenCC('t2s.json')


def backup(file, bak_file):
    if not os.path.exists(file):
        print(f'{file} does not exist, skipping backup...')
        return

    if os.path.exists(bak_file):
        print(f'{bak_file} already exists, skipping backup to preserve original...')
        return

    shutil.copy2(file, bak_file)


def convert(file):
    if not os.path.exists(file):
        print(f'{file} does not exist, skipping convert...')
        return

    with open(file, 'r', encoding='utf-8') as f:
        converted = converter.convert(f.read())

    with open(file, 'w', encoding='utf-8') as f:
        f.write(converted)

    print(f'Converted {file}')


def main():
    files_to_convert = []
    for data_set in SETS:
        files_to_convert.append(f'data/{data_set}/zh.txt')

    files_to_convert.append('data/tokenizer/balanced.txt')
    files_to_convert.append('data/tokenizer/imbalanced_10-1-1.txt')

    for file in files_to_convert:
        backup_file = file.replace('.txt', '_bak.txt')
        backup(file, backup_file)
        convert(file)


if __name__ == "__main__":
    main()
