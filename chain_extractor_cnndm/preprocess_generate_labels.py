import argparse
import codecs
import json
import numpy as np
import re
# Get a counter for the iterations
from tqdm import tqdm
import os
from make_tokenized_files import get_art_abs

#tqdm.monitor_interval = 0
from collections import Counter

import spacy
import neuralcoref

nlp = spacy.load("en_core_web_lg")
neuralcoref.add_to_pipe(nlp, max_dist=1000)

def compile_substring(start, end, split):
    if start == end:
        return split[start]
    return " ".join(split[start:end + 1])


def format_json(s):
    return json.dumps({'sentence': s}) + "\n"


def splits(s, num=200):
    return s.split()


def make_BIO_tgt(s, t):
    # tsplit = t.split()
    ssplit = s.split()
    startix = 0
    endix = 0
    matches = []
    matchstrings = Counter()
    while endix < len(ssplit):
        # last check is to make sure that phrases at end can be copied
        searchstring = compile_substring(startix, endix, ssplit)
        if searchstring in t \
                and endix < len(ssplit) - 1:
            endix += 1
        else:
            # only phrases, not words
            # uncomment the -1 if you only want phrases > len 1
            if startix >= endix:  # -1:
                matches.extend(["0"] * (endix - startix + 1))
                endix += 1
            else:
                # First one has to be 2 if you want phrases not words
                full_string = compile_substring(startix, endix - 1, ssplit)
                if matchstrings[full_string] >= 1:
                    matches.extend(["0"] * (endix - startix))
                else:
                    matches.extend(["1"] * (endix - startix))
                    matchstrings[full_string] += 1
                # endix += 1
            startix = endix
    edited_matches = []
    for word, tag in zip(ssplit, matches):
        if word == '<split1>':
            edited_matches.append('<split1>')
        else:
            edited_matches.append(tag)
    # print(ssplit)
    # print(edited_matches)
    # print(len(ssplit) == len(edited_matches))
    # exit(0)
    return " ".join(edited_matches)

# def get_heuristic_ner_chains(article, abstract):
#     sentences = article.split("<split1>")
#     sentences_mentions = []
#     for sent in sentences :
#         sentences_mentions.append([])
#         doc = nlp(sent)
#         for ent in doc.ents:
#             sentences_mentions[-1].extend(ent.text.lower().split(" "))
#     ent_tracker = {}
#     list_sent_arcs = []
#     for idx, sent in enumerate(sentences_mentions):
#         for ent in sent:
#             if ent in ent_tracker:
#                 for parent_idx in ent_tracker[ent]:
#                    #print(ent, idx, ent_tracker[ent])
#                     list_sent_arcs.append((idx, parent_idx))
#         for ent in sent:
#             if ent in ent_tracker:
#                 ent_tracker[ent].append(idx)
#             else:
#                 ent_tracker[ent] = [idx]
#     return str(list_sent_arcs)

def get_heuristic_ner_coref_chains(article, abstract):
    sent_len = [len(nlp(x)) for x in article.split('<split1>')]
    sentence_delim = [sum(sent_len[:i+1]) for i in range(len(sent_len))]
    nb_sentences = len(sentence_delim)
    doc = nlp(article.replace('<split1>', ' '))
    list_sent_arcs = {'coref':[], 'ner':[]}
    for idx, cluster in enumerate(doc._.coref_clusters):
        sentence_with_mention = []
        #print('cluster: ', cluster.main) # gives the representative of the cluster
        for mention in cluster.mentions:
            sentence_index = sentence_delim.index(min(i for i in sentence_delim if i > mention.end))
            #print('mention: ', mention, mention.start, mention.end, sentence_index)
            # Not counting references within a sentence.
            if sentence_index not in sentence_with_mention:
                if len(sentence_with_mention) != 0:
                    for prev_sent in sentence_with_mention:
                        #print(mention, prev_sent, sentence_index)
                        list_sent_arcs['coref'].append({'cluster_idx':idx, 'mention_text':mention.text, 'head_id':prev_sent, 'tail_id':sentence_index})
                sentence_with_mention.append(sentence_index)

    ent_tracker = {}
    for idx, ent in enumerate(doc.ents):
        sentence_index = sentence_delim.index(min(i for i in sentence_delim if i > ent.end))
        #print('ent: ', ent, ent.start, ent.end, sentence_index)
        # Not counting references within a sentence.
        words = ent.text.split(" ")
        for word in words:
            if word in ent_tracker and sentence_index not in ent_tracker[word]:
                for prev_sent in ent_tracker[word]:
                    #print(word, prev_sent, sentence_index)
                    list_sent_arcs['ner'].append({'entity_idx':idx, 'entity_text':ent.text, 'head_id':prev_sent, 'tail_id':sentence_index})
                for word in words:
                    if word in ent_tracker:
                        ent_tracker[word].append(sentence_index)
                    else:
                        ent_tracker[word] = [sentence_index]
                break
            if word not in ent_tracker:
                ent_tracker[word] = [sentence_index]
    #print(list_sent_arcs)
    return str(list_sent_arcs)

def process_content_sel_labels(article, abstract):
    ssplit = splits(article)
    # Skip empty lines
    if len(ssplit) < 2 or len(abstract.split()) < 2:
        return None
    # Build the target
    tgt = make_BIO_tgt(ssplit, abstract)
    return tgt

def process_heuristic_chain_labels(article, abstract):
    ssplit = splits(article)
    # Skip empty lines
    if len(ssplit) < 2 or len(abstract.split()) < 2:
        return None
    # Build the target
    chains = get_heuristic_ner_coref_chains(article, abstract)
    return chains

def write_labels(ner_out_dir, contsel_out_dir, stories, stories_dir):
    for s in tqdm(stories):
        in_path = os.path.join(stories_dir, s)
        ner_out_path = os.path.join(ner_out_dir, s)
        contsel_out_path = os.path.join(contsel_out_dir, s)
        article, abstract = get_art_abs(in_path)

        links = process_heuristic_chain_labels(article, abstract)
        if links is None:
            print(s, links)
            links = ""
        fp = open(ner_out_path, 'w')
        fp.write(links)
        fp.close()

        tags = make_BIO_tgt(article, abstract)
        if tags is None:
            print(s, tags)
            tags = ""
        fp = open(contsel_out_path, 'w')
        fp.write(tags)
        fp.close()


def main():
    # Directory names for input and output directories.

    parser = argparse.ArgumentParser(description='Script to preprocess tokenized files')
    parser.add_argument('--input_dir', type=str, default=None, help='location of the input dir')
    parser.add_argument('--ner_output_dir', type=str, default=None, help='location of the output dir')
    parser.add_argument('--contsel_output_dir', type=str, default=None, help='location of the output dir')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print("Folder: "+args.input_dir+" doesn't exist")
        exit()

    # cnn_stories_dir = '../cnn_stories_tokenized'
    # cnn_ner_label_dir = '../cnn_stories_ner_heuristic_chain_labels'
    # cnn_contsel_tags_label_dir =  '../cnn_stories_contsel_tags_labels'
    # dm_stories_dir = '../dm_stories_tokenized'
    # dm_ner_label_dir = '../dm_stories_ner_heuristic_chain_labels'
    # dm_contsel_tags_label_dir = '../dm_stories_contsel_tags_labels'
    #
    # if not os.path.exists(cnn_ner_label_dir):
    #    print("Creating cnn dir: ", cnn_ner_label_dir)
    #    os.mkdir(cnn_ner_label_dir)
    #
    # if not os.path.exists(cnn_contsel_tags_label_dir):
    #     print("Creating cnn dir: ", cnn_contsel_tags_label_dir)
    #     os.mkdir(cnn_contsel_tags_label_dir)
    #
    # if not os.path.exists(dm_ner_label_dir):
    #    print("Creating DM dir: ", dm_ner_label_dir)
    #    os.mkdir(dm_ner_label_dir)
    #
    # if not os.path.exists(dm_contsel_tags_label_dir):
    #     print("Creating DM dir: ", dm_contsel_tags_label_dir)
    #     os.mkdir(dm_contsel_tags_label_dir)

    # Write all labels labels into new dirs
    stories_dir = args.input_dir
    ner_out_dir = args.ner_output_dir
    contsel_out_dir = args.contsel_output_dir
    stories = os.listdir(stories_dir)
    write_labels(ner_out_dir, contsel_out_dir, stories, stories_dir)

    # # Write all dm labels to new dir
    # stories_dir = dm_stories_dir
    # ner_out_dir = dm_ner_label_dir
    # contsel_out_dir = cnn_contsel_tags_label_dir
    # stories = os.listdir(stories_dir)
    # write_labels(ner_out_dir, contsel_out_dir, stories, stories_dir)


if __name__ == "__main__":
    main()
