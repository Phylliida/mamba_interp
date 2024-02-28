from collections import defaultdict
import torch
import random
import re
import multiprocessing
import numpy as np
import copy
import itertools

from transformers import AutoTokenizer

from docstring import docstring_prompt_generator_function

MAMBA_TOKENIZER_STR = 'EleutherAI/gpt-neox-20b'

# my computer can't run 'mamba-2.8b', 'mamba-1.4b'? 'mamba-790m',?
MODELS = [ 'mamba-370m', 'mamba-130m']
MODELS = [('state-spaces/' + m) for m in MODELS]

def load(model_str, device='cuda', dtype=torch.float16, tokenizer_str=MAMBA_TOKENIZER_STR):
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    model = MambaLMHeadModel.from_pretrained(model_str, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_str)
    return model, tokenizer





######## Induction ##############


# todo: symmetric? probably just substitute a different random token
def copy_generator(tokenizer, num_examples, copy_seq_len, seed=27):
    random.seed(27)
    bos_token = tokenizer.bos_token_id
    torch.random.manual_seed(seed)
    first_len = None
    for i in range(num_examples):
        # generate one twice as big, it'll be messed up when decoding so that way we can clip it at desired token len
        while True:
            data = torch.randint(low=1000, high=tokenizer.vocab_size-1000, size=(copy_seq_len*2,))
            prompt = tokenizer.decode(data).encode("ascii", "ignore").decode("ascii", "ignore")
            tokens = tokenizer.encode(prompt)[:copy_seq_len]
            answer = tokenizer.decode([tokens[-1]])
            full_tokens = tokens + tokens[:-1]
            full_prompt = tokenizer.decode(full_tokens)
            full_tokens = tokenizer.encode(full_prompt)
            full_len = len(full_tokens)
            if first_len is None:
                first_len = full_len
            # retry until we match the size
            if not full_len == first_len:
                continue
            else:
                yield full_prompt, [answer], [tokenizer.pad_token]
                break




######### Greater than ###############

NOUNS = [
    "abduction", "accord", "affair", "agreement", "appraisal",
    "assaults", "assessment", "attack", "attempts", "campaign", 
    "captivity", "case", "challenge", "chaos", "clash", 
    "collaboration", "coma", "competition", "confrontation", "consequence", 
    "conspiracy", "construction", "consultation", "contact",
    "contract", "convention", "cooperation", "custody", "deal", 
    "decline", "decrease", "demonstrations", "development", "disagreement", 
    "disorder", "dispute", "domination", "dynasty", "effect", 
    "effort", "employment", "endeavor", "engagement",
    "epidemic", "evaluation", "exchange", "existence", "expansion", 
    "expedition", "experiments", "fall", "fame", "flights",
    "friendship", "growth", "hardship", "hostility", "illness", 
    "impact", "imprisonment", "improvement", "incarceration",
    "increase", "insurgency", "invasion", "investigation", "journey", 
    "kingdom", "marriage", "modernization", "negotiation",
    "notoriety", "obstruction", "operation", "order", "outbreak", 
    "outcome", "overhaul", "patrols", "pilgrimage", "plague",
    "plan", "practice", "process", "program", "progress", 
    "project", "pursuit", "quest", "raids", "reforms", 
    "reign", "relationship",
    "retaliation", "riot", "rise", "rivalry", "romance", 
    "rule", "sanctions", "shift", "siege", "slump", 
    "stature", "stint", "strikes", "study",
    "test", "testing", "tests", "therapy", "tour", 
    "tradition", "treaty", "trial", "trip", "unemployment", 
    "voyage", "warfare", "work",
]

# modified from ACDC https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/greaterthan/utils.py
def greater_than_data_generator(tokenizer, num_examples, seed=27):
    YEARS = []
    YEARS_CENTURY = defaultdict(lambda: [])

    for century in range(11, 18):
        all_success = []
        for year in range(century * 100 + 2, (century * 100) + 99):
            a = tokenizer.encode(f" {year}")
            # make sure it tokenizes cleanly into like 1420 -> 14 and 20
            if a == [tokenizer.encode(f" {str(year)[:2]}")[0], tokenizer.encode(str(year)[2:])[0]]:
                all_success.append(str(year))
        YEARS.extend(all_success[1:-1]) # this is to prevent stuff like 1999 (next year is a different century), that way we can just complete 19__
        YEARS_CENTURY[century].extend(all_success)

    nouns = restrict_to_most_common_size(tokenizer=tokenizer, words=NOUNS, with_space=True)
    print("nouns using", nouns)
    # set some random seed
    torch.random.manual_seed(seed)
    random.seed(seed)
    nouns_perm = torch.randint(0, len(nouns), (num_examples,))
    years_perm = torch.randint(0, len(YEARS), (num_examples,))
    
    for i in range(num_examples):
        year = YEARS[years_perm[i]]
        century, decade = int(year[:2]), int(year[2:])
        correct_outputs = []
        incorrect_outputs = []
        for output_year in YEARS_CENTURY[century]:
            output_century, output_decade = int(output_year[:2]), int(output_year[2:])
            if output_decade > decade:
                correct_outputs.append(str(output_decade))
            else:
                incorrect_outputs.append(str(output_decade))
        prompt = "The {noun} lasted from the year {year1} to ".format(
            noun=nouns[nouns_perm[i]],
            year1=year,
        ) + str(century) # first two tokens of year: like 1920 -> 19
        yield prompt, correct_outputs, incorrect_outputs 
            
##########################################################            
            

########################## IOI ###########################

# modified from https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/main/acdc/ioi/ioi_dataset.py

NAMES = [
    "Michael",
    "Christopher",
    "Jessica",
    "Matthew",
    "Ashley",
    "Jennifer",
    "Joshua",
    "Amanda",
    "Daniel",
    "David",
    "James",
    "Robert",
    "John",
    "Joseph",
    "Andrew",
    "Ryan",
    "Brandon",
    "Jason",
    "Justin",
    "Sarah",
    "William",
    "Jonathan",
    "Stephanie",
    "Brian",
    "Nicole",
    "Nicholas",
    "Anthony",
    "Heather",
    "Eric",
    "Elizabeth",
    "Adam",
    "Megan",
    "Melissa",
    "Kevin",
    "Steven",
    "Thomas",
    "Timothy",
    "Christina",
    "Kyle",
    "Rachel",
    "Laura",
    "Lauren",
    "Amber",
    "Brittany",
    "Danielle",
    "Richard",
    "Kimberly",
    "Jeffrey",
    "Amy",
    "Crystal",
    "Michelle",
    "Tiffany",
    "Jeremy",
    "Benjamin",
    "Mark",
    "Emily",
    "Aaron",
    "Charles",
    "Rebecca",
    "Jacob",
    "Stephen",
    "Patrick",
    "Sean",
    "Erin",
    "Jamie",
    "Kelly",
    "Samantha",
    "Nathan",
    "Sara",
    "Dustin",
    "Paul",
    "Angela",
    "Tyler",
    "Scott",
    "Katherine",
    "Andrea",
    "Gregory",
    "Erica",
    "Mary",
    "Travis",
    "Lisa",
    "Kenneth",
    "Bryan",
    "Lindsey",
    "Kristen",
    "Jose",
    "Alexander",
    "Jesse",
    "Katie",
    "Lindsay",
    "Shannon",
    "Vanessa",
    "Courtney",
    "Christine",
    "Alicia",
    "Cody",
    "Allison",
    "Bradley",
    "Samuel",
]

ABC_TEMPLATES = [
    "Then, [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "Afterwards [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "When [A], [B] and [C] arrived at the [PLACE], [B] and [C] gave a [OBJECT] to [A]",
    "Friends [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
]

BAC_TEMPLATES = [
    template.replace("[B]", "[A]", 1).replace("[A]", "[B]", 1)
    for template in ABC_TEMPLATES
]

BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BABA_LONG_TEMPLATES = [
    "Then in the morning, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After taking a long break [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While spending time together [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While spending time together [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch in the afternoon, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, while spending time together [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning afterwards, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The local big [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends separated at birth [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BABA_LATE_IOS = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument and after that [B] said to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
]

BABA_EARLY_IOS = [
    "Then [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] had a lot of fun at the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] were working at the [PLACE], and [B] decided to give a [OBJECT] to [A]",
    "Then [B] and [A] were thinking about going to the [PLACE], and [B] wanted to give a [OBJECT] to [A]",
    "Then [B] and [A] had a long argument, and after that [B] said to [A]",
    "After the lunch [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Afterwards [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] had a long argument, and afterwards [B] said to [A]",
]

TEMPLATES_VARIED_MIDDLE = [
    "",
]

# no end of texts, GPT-2 small wasn't trained this way (ask Arthur)
# warnings.warn("Adding end of text prefixes!")
# for TEMPLATES in [BABA_TEMPLATES, BABA_EARLY_IOS, BABA_LATE_IOS]:
#     for i in range(len(TEMPLATES)):
#         TEMPLATES[i] = "<|endoftext|>" + TEMPLATES[i]

ABBA_TEMPLATES = BABA_TEMPLATES[:]
ABBA_LATE_IOS = BABA_LATE_IOS[:]
ABBA_EARLY_IOS = BABA_EARLY_IOS[:]

for TEMPLATES in [ABBA_TEMPLATES, ABBA_LATE_IOS, ABBA_EARLY_IOS]:
    for i in range(len(TEMPLATES)):
        first_clause = True
        for j in range(1, len(TEMPLATES[i]) - 1):
            if TEMPLATES[i][j - 1 : j + 2] == "[B]" and first_clause:
                TEMPLATES[i] = TEMPLATES[i][:j] + "A" + TEMPLATES[i][j + 1 :]
            elif TEMPLATES[i][j - 1 : j + 2] == "[A]" and first_clause:
                first_clause = False
                TEMPLATES[i] = TEMPLATES[i][:j] + "B" + TEMPLATES[i][j + 1 :]

VERBS = [" tried", " said", " decided", " wanted", " gave"]
PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]
OBJECTS = [
    "apple", # ring confuses the model
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]

ANIMALS = [
    "dog",
    "cat",
    "snake",
    "elephant",
    "beetle",
    "hippo",
    "giraffe",
    "tiger",
    "husky",
    "lion",
    "panther",
    "whale",
    "dolphin",
    "beaver",
    "rabbit",
    "fox",
    "lamb",
    "ferret",
]

NOUNS_DICT = {"[PLACE]": PLACES, "[OBJECT]": OBJECTS}

# from https://stackoverflow.com/a/2556252
# replaces the last occurance instances of old with new
# (so if occurance is 1, it replaces the last instance of old with new)
# (if occurance is 2, it replaces the last two instances of old with new)
# etc.
def replace_n_last_occurance(s, old, new, n):
    li = s.rsplit(old, n)
    return new.join(li)



PREFIXES = [
    "             Afterwards,",
    "            Two friends met at a bar. Then,",
    "  After a long day,",
    "  After a long day,",
    "    Then,",
    "         Then,",
]

def flip_prefixes(ioi_prompts):
    ioi_prompts = copy.deepcopy(ioi_prompts)
    for prompt in ioi_prompts:
        if prompt["text"].startswith("The "):
            prompt["text"] = "After the lunch, the" + prompt["text"][4:]
        else:
            io_idx = prompt["text"].index(prompt["IO"])
            s_idx = prompt["text"].index(prompt["S"])
            first_idx = min(io_idx, s_idx)
            prompt["text"] = random.choice(PREFIXES) + " " + prompt["text"][first_idx:]

    return ioi_prompts

def gen_flipped_prompts(prompts, names, animals, flip, seed=None):
    """
    Return a IOIDataset where the name to flip has been replaced by a random name.
    """

    assert seed is not None

    assert isinstance(flip, tuple) or flip in [
        "prefix",
    ], f"{flip} is not a tuple. Probably change to ('IO', 'RAND') or equivalent?"

    if flip == "prefix":
        flipped_prompts = flip_prefixes(prompts)
    else:
        if flip in [("IO", "S1"), ("S", "IO")]:
            flipped_prompts = gen_flipped_prompts_helper(
                prompts=prompts,
                names=names,
                animals=animals,
                flip=flip,
                seed=(seed+12345)%9876,
            )
        elif flip == ("S2", "IO"):
            flipped_prompts = gen_flipped_prompts_helper(
                prompts=prompts,
                names=names,
                animals=animals,
                flip=flip,
                seed=(seed+12345)%6543,
            )

        else:
            assert flip[1] == "RAND" and flip[0] in [
                "S",
                "RAND",
                "S2",
                "IO",
                "S1",
                "S+1",
            ], flip
            flipped_prompts = gen_flipped_prompts_helper(prompts=prompts, names=names, animals=animals, flip=flip, seed=(seed+345467)%5432)
    return flipped_prompts

def gen_flipped_prompts_helper(prompts, names, animals, flip=("S2", "IO"), seed=None):
    """_summary_

    Args:
        prompts (List[D]): _description_
        flip (tuple, optional): First element is the string to be replaced, Second is what to replace with. Defaults to ("S2", "IO").

    Returns:
        _type_: _description_
    """

    assert seed is not None
    np.random.seed(seed)

    flipped_prompts = []

    for prompt in prompts:
        t = prompt["text"].split(" ")
        prompt = prompt.copy()
        if flip[0] == "S2":
            if flip[1] == "IO":
                t[len(t) - t[::-1].index(prompt["S"]) - 1] = prompt["IO"]
                temp = prompt["IO"]
                prompt["IO"] = prompt["S"]
                prompt["S"] = temp
            elif flip[1] == "RAND":
                rand_name = names[np.random.randint(len(names))]
                while rand_name == prompt["IO"] or rand_name == prompt["S"]:
                    rand_name = names[np.random.randint(len(names))]
                t[len(t) - t[::-1].index(prompt["S"]) - 1] = rand_name
            else:
                raise ValueError("Invalid flip[1] value")

        elif flip[0] == "IO":
            if flip[1] == "RAND":
                rand_name = names[np.random.randint(len(names))]
                while rand_name == prompt["IO"] or rand_name == prompt["S"]:
                    rand_name = names[np.random.randint(len(names))]

                t[t.index(prompt["IO"])] = rand_name
                t[t.index(prompt["IO"])] = rand_name
                prompt["IO"] = rand_name
            elif flip[1] == "ANIMAL":
                rand_animal = animals[np.random.randint(len(animals))]
                t[t.index(prompt["IO"])] = rand_animal
                prompt["IO"] = rand_animal
                # print(t)
            elif flip[1] == "S1":
                io_index = t.index(prompt["IO"])
                s1_index = t.index(prompt["S"])
                io = t[io_index]
                s1 = t[s1_index]
                t[io_index] = s1
                t[s1_index] = io
            else:
                raise ValueError("Invalid flip[1] value")

        elif flip[0] in ["S", "S1"]:
            if flip[1] == "ANIMAL":
                new_s = animals[np.random.randint(len(animals))]
            if flip[1] == "RAND":
                new_s = names[np.random.randint(len(names))]
            t[t.index(prompt["S"])] = new_s
            if flip[0] == "S":  # literally just change the first S if this is S1
                t[len(t) - t[::-1].index(prompt["S"]) - 1] = new_s
                prompt["S"] = new_s
        elif flip[0] == "END":
            if flip[1] == "S":
                t[len(t) - t[::-1].index(prompt["IO"]) - 1] = prompt["S"]
        elif flip[0] == "PUNC":
            n = []

            # separate the punctuation from the words
            for i, word in enumerate(t):
                if "." in word:
                    n.append(word[:-1])
                    n.append(".")
                elif "," in word:
                    n.append(word[:-1])
                    n.append(",")
                else:
                    n.append(word)

            # remove punctuation, important that you check for period first
            if flip[1] == "NONE":
                if "." in n:
                    n[n.index(".")] = ""
                elif "," in n:
                    n[len(n) - n[::-1].index(",") - 1] = ""

            # remove empty strings
            while "" in n:
                n.remove("")

            # add punctuation back to the word before it
            while "," in n:
                n[n.index(",") - 1] += ","
                n.remove(",")

            while "." in n:
                n[n.index(".") - 1] += "."
                n.remove(".")

            t = n

        elif flip[0] == "C2":
            if flip[1] == "A":
                t[len(t) - t[::-1].index(prompt["C"]) - 1] = prompt["A"]
        elif flip[0] == "S+1":
            if t[t.index(prompt["S"]) + 1] == "and":
                t[t.index(prompt["S"]) + 1] = [
                    "with one friend named",
                    "accompanied by",
                ][np.random.randint(2)]
            else:
                t[t.index(prompt["S"]) + 1] = (
                    t[t.index(prompt["S"])]
                    + ", after a great day, "
                    + t[t.index(prompt["S"]) + 1]
                )
                del t[t.index(prompt["S"])]
        else:
            raise ValueError(f"Invalid flipper {flip[0]}")

        if "IO" in prompt:
            prompt["text"] = " ".join(t)
            flipped_prompts.append(prompt)
        else:
            flipped_prompts.append(
                {
                    "A": prompt["A"],
                    "B": prompt["B"],
                    "C": prompt["C"],
                    "text": " ".join(t),
                }
            )

    return flipped_prompts



def gen_prompt_uniform(
    templates, names, nouns_dict, N, symmetric, prefixes=None, abc=False, seed=None,
):
    if seed is not None:
        random.seed(seed)

    nb_gen = 0
    ioi_prompts = []
    while nb_gen < N:
        temp = random.choice(templates)
        temp_id = templates.index(temp)
        name_1 = ""
        name_2 = ""
        name_3 = ""
        while len(set([name_1, name_2, name_3])) < 3:
            name_1 = random.choice(names)
            name_2 = random.choice(names)
            name_3 = random.choice(names)

        nouns = {}
        ioi_prompt = {}
        for k in nouns_dict:
            nouns[k] = random.choice(nouns_dict[k])
            ioi_prompt[k] = nouns[k]
        prompt = temp
        for k in nouns_dict:
            prompt = prompt.replace(k, nouns[k])

        if prefixes is not None:
            L = random.randint(30, 40)
            pref = ".".join(random.choice(prefixes).split(".")[:L])
            pref += "<|endoftext|>"
        else:
            pref = ""

        prompt1 = prompt.replace("[A]", name_1)
        prompt1 = prompt1.replace("[B]", name_2)
        if abc:
            prompt1 = prompt1.replace("[C]", name_3)
        prompt1 = pref + prompt1
        ioi_prompt["text"] = prompt1
        ioi_prompt["IO"] = name_1
        ioi_prompt["S"] = name_2
        ioi_prompt["TEMPLATE_IDX"] = temp_id
        ioi_prompts.append(ioi_prompt)
        if abc:
            ioi_prompts[-1]["C"] = name_3

        nb_gen += 1

        if symmetric and nb_gen < N:
            # note this is modified from the original, we just replace the last occurance which makes them swapped
            prompt2 = replace_n_last_occurance(prompt1, name_2, name_1, 1)
            prompt2 = replace_n_last_occurance(prompt2, name_1, name_2, 1)
            prompt2 = pref + prompt2
            ioi_prompts.append(
                {"text": prompt2, "IO": name_2, "S": name_1, "TEMPLATE_IDX": temp_id}
            )
            if abc:
                ioi_prompts[-1]["C"] = name_3
            nb_gen += 1
    return ioi_prompts



def strip_to_first_token(tokenizer, s):
    res = tokenizer.decode([tokenizer.encode(s)[0]])
    if res.strip() == "":
        raise Exception(f"when turned into single token {s} becomes only whitespace")
    return res

def all_abc():
    abc_formats = ['ABC DEF']
    share_one_abc = ['ABC', 'BAC', 'BCA']
    share_one_ade = [x.replace("B", "D").replace("C", "E") for x in share_one_abc]
    abc_formats += [a + " " + b for (a,b) in itertools.product(share_one_abc, share_one_ade)]
    
    # share two
    share_two_abc = ['ABC', 'ACB', 'CAB']
    share_two_ade = [x.replace("C", "D") for x in share_two_abc]
    share_two_all = [(a,b) for (a,b) in itertools.product(share_two_abc, share_two_ade)]
    abc_formats += [a + " " + b for a,b in share_two_all]
    # swap A and B in second one
    abc_formats += [a + " " + b.replace("A", "W").replace("B", "A").replace("W","B") for a,b in share_two_all]
    
    # share three
    abc_formats += ['ABC ABC']
    abc_formats += ['ABC ACB', 'ABC CBA', 'ABC BAC'] # fix one, swap others
    abc_formats += ['ABC BCA', 'ABC CAB'] # shift
    for format in all_formats(abc_formats):
        yield format

def all_possible_postfixes(abc):
    tokens = sorted(list(set(abc)))
    for output1, output2 in itertools.combinations(tokens, 2):
        other = set(tokens)
        other.remove(output1)
        other.remove(output2)
        other = list(other)[0]
        yield f"{abc} {output1}{output2} {other}"

def all_formats(abc_formats):
    for s in abc_formats:
        first_abc, second_abc = s.split()
        for s1 in all_possible_postfixes(first_abc):
            for s2 in all_possible_postfixes(second_abc):
                yield s1 + "\n" + s2

def IOI_custom_generator(ioi_format, tokenizer, num_examples, seed):

    global good_names
    global good_nouns
    if not 'good_names' in globals() or good_names is None:
        with open("first-names.txt", "r") as f:
            names = [x.strip() for x in f.read().split("\n") if len(x.strip()) > 0]
            
        names = restrict_to_most_common_size(tokenizer, names, with_space=True, force_size=1)

        # stuff like "chris" and "christine" get confused, ignore them
        no_prefix_names = []
        for name in names:
            has_other_as_prefix = any([other.startswith(name) and other != name for other in names])
            if not has_other_as_prefix:
                no_prefix_names.append(name)
            else:
                other_prefix = names[[other.startswith(name) and other != name for other in names].index(True)]
        names = no_prefix_names
        good_names = names

        noun_dict = {}
        for k,v in NOUNS_DICT.items():
            noun_dict[k] = restrict_to_most_common_size(tokenizer, v, with_space=True)
        good_nouns = noun_dict
        print("good nouns", good_nouns)
    try:
        prompt_uncorrupted_template = 'Lately, [A], [B], and [C] had fun at the [PLACE]. [D] and [E] gave a [OBJECT] to'
        prompt_corrupted_template = 'Lately, [F], [G], and [H] had fun at the [PLACE]. [I] and [J] gave a [OBJECT] to'
        
        (a,b,c,space,d,e,space,answer1),(f,g,h,space,i,j,space,answer2) = ioi_format.split("\n")
    except:
        prompt_uncorrupted_template = 'Lately, [A] and [B] had fun at the [PLACE]. [D] gave a [OBJECT] to'
        prompt_corrupted_template = 'Lately, [F] and [G] had fun at the [PLACE]. [I] gave a [OBJECT] to'
        (a,b,space,d,space,answer1),(f,g,space,i,space,answer2) = ioi_format.split("\n")
        c,e,h,j = a,a,a,a

    random.seed(seed)
    good_names = sorted(good_names)
    
    unique_tokens = set(re.sub(r"\s*", "", ioi_format))
    for n in range(num_examples//2):
        random.shuffle(good_names)
        tok_map = {}
        for ind, tok in enumerate(unique_tokens):
            tok_map[tok] = good_names[ind]
        place = random.choice(good_nouns['[PLACE]'])
        object = random.choice(good_nouns['[OBJECT]'])
        def replace_things(text, tok_map):
            text = text.replace('[A]', tok_map[a])
            text = text.replace('[B]', tok_map[b])
            text = text.replace('[C]', tok_map[c])
            text = text.replace('[D]', tok_map[d])
            text = text.replace('[E]', tok_map[e])
            text = text.replace('[F]', tok_map[f])
            text = text.replace('[G]', tok_map[g])
            text = text.replace('[H]', tok_map[h])
            text = text.replace('[I]', tok_map[i])
            text = text.replace('[J]', tok_map[j])
            text = text.replace("[PLACE]", place)
            text = text.replace("[OBJECT]", object)
            return text
        all_answers = [' ' + x for x in tok_map.values()]
        prompt_uncorrupted = replace_things(prompt_uncorrupted_template, tok_map)
        prompt_corrupted = replace_things(prompt_corrupted_template, tok_map)
        uncorrupted_answer = ' ' + tok_map[answer1]
        corrupted_answer = ' ' + tok_map[answer2]
        uncorrupted_incorrect = list(all_answers)
        uncorrupted_incorrect.remove(uncorrupted_answer)
        corrupted_incorrect = list(all_answers)
        corrupted_incorrect.remove(corrupted_answer)
        yield prompt_uncorrupted, [uncorrupted_answer], uncorrupted_incorrect
        yield prompt_corrupted, [corrupted_answer], corrupted_incorrect


def IOI_generator(tokenizer, num_examples, seed=27, templates=None, symmetric=True):
    abc = False
    for template in templates:
        if '[C]' in template:
            abc = True
            
            
    # make sure they are the same number of tokens so interventions line up
    noun_dict = {}
    for k,v in NOUNS_DICT.items():
        noun_dict[k] = restrict_to_most_common_size(tokenizer, v, with_space=True)
    
    with open("first-names.txt", "r") as f:
        names = [x.strip() for x in f.read().split("\n") if len(x.strip()) > 0]
        
    names = restrict_to_most_common_size(tokenizer, names, with_space=True, force_size=1)

    # stuff like "chris" and "christine" get confused, ignore them
    no_prefix_names = []
    for name in names:
        has_other_as_prefix = any([other.startswith(name) and other != name for other in names])
        if not has_other_as_prefix:
            no_prefix_names.append(name)
        else:
            other_prefix = names[[other.startswith(name) and other != name for other in names].index(True)]
    names = no_prefix_names
    
    #print("nouns", noun_dict)
    
    #print("names", names)
            
    prompts = gen_prompt_uniform(templates=templates, names=names, nouns_dict=noun_dict, N=num_examples, symmetric=symmetric, abc=abc, seed=seed)
    for prompt in prompts:
        indirect_object = prompt['IO']
        #print("prompt", prompt['text'])
        #print("io", indirect_object)
        prompt_text = prompt['text'].strip()[:-len(indirect_object)-1] # -1 for space
        # the strip_to_first_token is necessary because some names are multiple tokens, we only care about first token output
        correct = [strip_to_first_token(tokenizer, " "  + prompt['IO'])] # space before is important so it is a single token
        incorrect = [strip_to_first_token(tokenizer, " "  + prompt['S'])]
        if abc and prompt['C'] in prompt_text:
            incorrect.append(strip_to_first_token(tokenizer, " "  + prompt['C']))
        yield prompt_text, correct, incorrect





###################################################################



# restricts words to only words with the same size tokens
# it choses which size to use based on whichever is most common among the words
# if with_space is true, it considers tokenization when a space is added in front of the word
def restrict_to_most_common_size(tokenizer, words, with_space=False, force_size=None):
    sizes = defaultdict(lambda: 0)
    
    if with_space:
        tokenized_words = [tokenizer.encode(" "  + word) for word in words]
    else:
        tokenized_words = [tokenizer.encode(word) for word in words]
    
    for toks in tokenized_words:
        sizes[len(toks)] += 1
    
    biggest_size, biggest_count = max(sizes.items(), key=lambda x: x[1])
    if force_size:
        biggest_size = force_size
    return [word for toks, word in zip(tokenized_words, words) if len(toks) == biggest_size]




def ndigits(n, f):
    return ("{:." + str(n) + "f}").format(f)

            

def print_data(num_correct, num_incorrect, num_other, correct_prompts, incorrect_prompts, other_prompts, print_fun=None):
    
    top_n = len(num_correct)
    
    if print_fun is None:
        print_fun = print
    
    print_fun("totals:")
    for i in range(top_n):
        total = num_correct[i] + num_incorrect[i] + num_other[i]
        if total != 0: # this can happen if top_n_toks ends up being smaller than top_n because there's only 2-3 possible outputs and we have constrain=True
            print_fun(f"  total for top_{i+1}:")
            print_fun(f"    +: {num_correct[i]} / {total} = {num_correct[i]/float(total)}")
            print_fun(f"    -: {num_incorrect[i]} / {total} = {num_incorrect[i]/float(total)}")
            print_fun(f"    o: {num_other[i]} / {total} = {num_other[i]/float(total)}")
    
    print_fun("\nfailures:")
    for i in range(top_n):
        if len(incorrect_prompts[i]) > 0:
            print_fun(f"  failures for top_{i+1}")
            for prompt, output, pr, relative_pr in incorrect_prompts[i]:
                print_fun(f"    {prompt}")
                print_fun(f"      {output} with pr {ndigits(4, pr)} relative pr {ndigits(4, relative_pr)}")
        if len(other_prompts[i]) > 0:
            print_fun(f"  other    for top_{i+1}")
            for prompt, output, pr, relative_pr in other_prompts[i]:
                print_fun(f"    {prompt}")
                print_fun(f"      {output} with pr {ndigits(4, pr)} relative pr {ndigits(4, relative_pr)}")
    


# constrain will limit generation to only correct and incorrect things provided by data_generator
def test_data(model, tokenizer, data_generator, num_examples, top_n=1, debug=False, constrain=True, device='cuda', *args, **kwargs):
    bos_token = tokenizer.bos_token_id
    
    num_correct = defaultdict(lambda: 0)
    num_incorrect = defaultdict(lambda: 0)
    num_other = defaultdict(lambda: 0)
    correct_prompts = defaultdict(lambda: [])
    incorrect_prompts = defaultdict(lambda: [])
    other_prompts = defaultdict(lambda: [])
    
    for i in range(top_n): # so its easy to get top_
        num_correct[i] = 0
        num_incorrect[i] = 0
        num_other[i] = 0
        correct_prompts[i] = []
        incorrect_prompts[i] = []
        other_prompts[i] = []

    
    for prompt, correct, incorrect in data_generator(tokenizer=tokenizer, num_examples=num_examples, *args, **kwargs):
        encoded_options = [tokenizer.encode(x) for x in (correct + incorrect)]
        for toks in encoded_options:
            if len(toks) > 1:
                raise Exception(f"for prompt {prompt} output '{tokenizer.decode(toks)}' is more than one token, specifically {toks}")
        all_valid = torch.tensor([toks[0] for toks in encoded_options])
        input = torch.tensor([bos_token] + tokenizer.encode(prompt)).to(device).view(1, -1)
        
        with torch.no_grad():
            logits = model(input)[0,-1] # 0 because first index is batch, -1 because logits of last token
            prs = torch.softmax(logits, dim=0)
            relative_prs = prs
        relative_prs = torch.softmax(logits[all_valid], dim=0)
        if constrain:
            prs = prs[all_valid]
            prs_toks = all_valid
        else:
            prs_toks = torch.arange(tokenizer.vocab_size)
        top_n_toks = torch.argsort(-prs)[:top_n]
        
        if debug:
            print(prompt)
        
            print(f" top {top_n}")
        for top_n_pos, i in enumerate(top_n_toks):
            tok = prs_toks[i]
            pr = prs[i].item()
            if constrain:
                relative_pr = relative_prs[i]
            else:
                lookup = (all_valid == tok).nonzero(as_tuple=True)[0]
                if lookup.size()[0] > 0:
                    relative_pr = relative_prs[lookup][0].item() # lookup where this token is in the relative_prs array
                else:
                    relative_pr = torch.nan
            pr = float(pr)
            relative_pr = float(relative_pr)
            s = tokenizer.decode([tok])
            if debug: print(f"  {s} pr {pr} relative_pr {relative_pr}")
            if s in correct:
                num_correct[top_n_pos] += 1
                correct_prompts[top_n_pos].append((prompt, s, pr, relative_pr))
                if debug: print("    +")
            elif s in incorrect:
                num_incorrect[top_n_pos] += 1
                incorrect_prompts[top_n_pos].append((prompt, s, pr, relative_pr))
                if debug: print("    -")
            else:
                num_other[top_n_pos] += 1
                other_prompts[top_n_pos].append((prompt, s, pr, relative_pr))
                if debug: print("    o")
    
    if debug:
        print_data(num_correct, num_incorrect, num_other, correct_prompts, incorrect_prompts, other_prompts)
    
    
    return dict(num_correct.items()), dict(num_incorrect.items()), dict(num_other.items()), dict(correct_prompts.items()), dict(incorrect_prompts.items()), dict(other_prompts.items())
    


## Hack where we spin up a seperate process each time we init a model
# this ensures it actually cleans up itself upon being done with it

def wrapper(process_queue, model_str, args, kwargs):
    model, tokenizer = load(model_str)
    kwargs['model'] = model
    kwargs['tokenizer'] = tokenizer
    generate_data(*args, **kwargs)
    process_queue.put("done")

def run_generate_data_seperate_process(model_str, *args, **kwargs):
    from multiprocessing import Process

    process_queue = multiprocessing.Queue()
    try:
        p = multiprocessing.Process(target=wrapper, args=(process_queue, model_str, args, kwargs), daemon=False)
        p.start()
        res = process_queue.get()
    except KeyboardInterrupt: # if we kill task, make sure cleanup model before raising
        p.terminate()
        raise
        pass
    p.terminate() # there is bugs in the mamba cleanup and it doesn't cleanup properly and hangs, so kill it since we are done
    return res
    
## Done hack

    
def generate_data_all_models(*args, **kwargs):
    for model_str in MODELS:
        print(f"-------- {model_str} --------")
        kwargs['output_file'] = model_str.replace("state-spaces/", "") + " test outputs.txt"
        run_generate_data_seperate_process(model_str=model_str, *args, **kwargs)
    
    
def generate_data(model, tokenizer, output_file, top_n=5, num_examples=300):

    print("-------- Greater Than --------")
    gt_data = {}
    gt_data = test_data(data_generator=greater_than_data_generator, model=model, tokenizer=tokenizer, num_examples=num_examples, top_n=top_n, constrain=False, debug=True)
     
    ioi_data = {}
    
    TEMPLATES = [
        ("ABC", ABC_TEMPLATES),
        ("BAC", BAC_TEMPLATES),
        ("BABA", BABA_TEMPLATES),
        ("BABA_LONG", BABA_LONG_TEMPLATES),
        ("BABA_LATE", BABA_LATE_IOS),
        ("BABA_EARLY", BABA_EARLY_IOS)
    ]
    
    print("\n\n-------- IOI --------")
    for template_name, templates in TEMPLATES:
        print(f"-------- IOI {template_name} --------")
        ioi_data[template_name] = test_data(data_generator=IOI_generator, templates=templates,  model=model, tokenizer=tokenizer, num_examples=num_examples, top_n=1, constrain=True, debug=True)
     
    print("\n\n-------- Docstring --------")
    docstring_data = {}
    docstring_data = test_data(data_generator=docstring_prompt_generator_function, model=model, tokenizer=tokenizer, num_examples=num_examples, top_n=1, constrain=False, debug=True)
     
    print("writing final results")
    
    with open(output_file, "w") as f:
        
        f.write("-------- Greater Than --------\n")
        print_data(*gt_data, print_fun=lambda s: f.write(s + "\n"))
        
        f.write("\n\n-------- IOI --------\n")
        for template_name, _ in TEMPLATES:
            f.write(f"-------- IOI {template_name} --------\n")
            print_data(*ioi_data[template_name], print_fun=lambda s: f.write(s + "\n"))
                
        f.write("\n\n-------- Docstring --------\n")
        print_data(*docstring_data, print_fun=lambda s: f.write(s + "\n"))
        
