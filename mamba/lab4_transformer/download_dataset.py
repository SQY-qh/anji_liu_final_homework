import os
import random

base_pairs = [
    ("ich liebe dich", "i love you"),
    ("guten morgen", "good morning"),
    ("guten abend", "good evening"),
    ("wie geht es dir", "how are you"),
    ("danke schön", "thank you"),
    ("bitte sehr", "you are welcome"),
    ("ich heiße anna", "my name is anna"),
    ("wo ist der bahnhof", "where is the station"),
    ("ich komme aus berlin", "i am from berlin"),
    ("ich spreche ein bisschen deutsch", "i speak a little german"),
    ("hast du zeit", "do you have time"),
    ("ich habe hunger", "i am hungry"),
    ("ich habe durst", "i am thirsty"),
    ("das ist gut", "that is good"),
    ("das ist schlecht", "that is bad"),
    ("ich verstehe nicht", "i do not understand"),
    ("kannst du helfen", "can you help"),
    ("wie spät ist es", "what time is it"),
    ("wo wohnst du", "where do you live"),
    ("bis später", "see you later"),
    ("bis morgen", "see you tomorrow"),
    ("entschuldigung", "excuse me"),
    ("es tut mir leid", "i am sorry"),
    ("viel glück", "good luck"),
    ("gute nacht", "good night"),
    ("was machst du", "what are you doing"),
    ("ich brauche hilfe", "i need help"),
    ("wo ist die toilette", "where is the toilet"),
    ("wie viel kostet das", "how much does it cost"),
    ("ein bier bitte", "a beer please"),
]

def augment(de, en):
    ops = []
    ops.append((de, en))
    ops.append((de.capitalize(), en.capitalize()))
    ops.append((de + ".", en + "."))
    ops.append((de.replace(" ", "  "), en))
    return ops

def build_dataset(train_size=100, test_size=10, seed=42):
    random.seed(seed)
    data = []
    for de, en in base_pairs:
        data.extend(augment(de, en))
    random.shuffle(data)
    data = data[: train_size + test_size]
    train = data[:train_size]
    test = data[train_size: train_size + test_size]
    return train, test

def write_split(split, path_de, path_en):
    with open(path_de, "w", encoding="utf-8") as fde:
        with open(path_en, "w", encoding="utf-8") as fen:
            for de, en in split:
                fde.write(de.strip() + "\n")
                fen.write(en.strip() + "\n")

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)
    train, test = build_dataset()
    write_split(train, os.path.join(data_dir, "train.de"), os.path.join(data_dir, "train.en"))
    write_split(test, os.path.join(data_dir, "test.de"), os.path.join(data_dir, "test.en"))
    print("dataset_ready", len(train), len(test))

if __name__ == "__main__":
    main()
