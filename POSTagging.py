###********************************************************************************###
  # __author__ = 'sid'                                                             #
  # This program is written as part of the Natural Language Processing Home Work 3 #
  # @copyright: Sudarshan Sudarshan (Sid)                                          #
###********************************************************************************###

import nltk
import math
import time
import operator
from itertools import izip
import re


class pos_training:
    @staticmethod
    def tokenization(fpath):

        pos = {}
        no_of_tags = 0
        word_tag = {}
        transition = {}

        starttags = ["<s>/<s>"]  # Dummy start symbol
        endtags = ["<e>/<e>"]  # Dummy end symbol

        file_content = open(fpath)

        for line in file_content.readlines():
            tokens = starttags + nltk.WhitespaceTokenizer().tokenize(line) + endtags

            for index, token in enumerate(tokens):  # Create the dictionary

                # Increment the No_of_tags by 1
                no_of_tags += 1

                # Add the <word tag: count> to dictionary
                word = token.split("/")[0]
                tag = token.split("/")[1]
                if word + " " + tag in word_tag:
                    word_tag[word + " " + tag] += 1
                else:
                    word_tag[word + " " + tag] = 1

                # Add the pos occurrence to dictionary
                if tag in pos:
                    pos[tag] += 1
                else:
                    pos[tag] = 1

                # Get the transition tags
                if index < len(tokens) - 1:
                    tag1 = tokens[index].split("/")[1]
                    tag2 = tokens[index + 1].split("/")[1]
                    if (tag1 + " " + tag2) in transition:
                        transition[tag1 + " " + tag2] += 1
                    else:
                        transition[tag1 + " " + tag2] = 1
        # tags dictionary, transition dictionary, word_tag dictionary, no of tags in the file
        token_results = [pos, transition, word_tag, no_of_tags]

        return token_results

    # Function to calculate the Unigram probability of tags
    @staticmethod
    def Unigram_Probability(pos, No_of_tags):

        # A dictionary to store the <unigram: probability>
        pos_probability = {}

        for word, count in pos.items():
            # Use MLE Estimation
            pos_probability[word] = (count / float(No_of_tags))

        return pos_probability

    # Function to calculate tag_tag probability
    @staticmethod
    def tagtag_probability(tagtag, pos, pos_prob):

        # For transition probability we must use the Interpolation smoothing technique
        alpha = 0.99
        beta = 1 - alpha

        tagtag_prob = {}
        for word, count in tagtag.items():
            # tagtag_prob[word] = (alpha * (count / float(pos[word.split(" ")[1]]))) + (
            #     beta * float(pos_prob[word.split(" ")[0]]))
            tagtag_prob[word] = (count / float(pos[word.split(" ")[0]]))

        return tagtag_prob

    # Function to calculate word_tag probability
    @staticmethod
    def wordtag_probability(wordtag, pos):

        wordtag_prob = {}
        for word, count in wordtag.items():
            wordtag_prob[word] = (count / float(pos[word.split(" ")[1]]))

        return wordtag_prob


class pos_testing:
    @staticmethod
    def read_lmfile(lmfile):

        # To get pos, transition probability and observation probability
        tagtag_probability = {}
        wordtag_probability = {}
        tags = {}

        lm_content = open(lmfile)
        for lNo, line in enumerate(lm_content.readlines()):
            if line == "pos:\n":
                posData = lNo
            if line == "transition:\n":
                transData = lNo
            elif line == "observation:\n":
                obsData = lNo

        lm_content.close()

        init_viterbi = {}
        seenWords = []

        lm_content = open(lmfile)
        for lNo, line in enumerate(lm_content.readlines()):
            # read tags
            if posData < lNo < transData:
                tags[line.split("\t")[0]] = float(line.split("\t")[1].rstrip('\n'))
            # read tagtag
            elif transData < lNo < obsData:
                tagtag_probability[line.split("\t")[0]] = float(line.split("\t")[1].rstrip('\n'))
                # get the initial viterbi
                if "<s>" == (line.split("\t")[0]).split(" ")[0]:
                    init_viterbi[line.split("\t")[0]] = float(line.split("\t")[1].rstrip('\n')) + tags[
                        line.split("\t")[0].split(" ")[0]]
            # read wordtag
            elif lNo > obsData:
                wordtag_probability[line.split("\t")[0]] = float(line.split("\t")[1].rstrip('\n'))
                seenWords.append(line.split("\t")[0].split(" ")[0])

        lm_model = [tags, tagtag_probability, wordtag_probability, init_viterbi, set(seenWords)]
        return lm_model

    @staticmethod
    def tag_testfile(testfile, tags, tran_prob, obs_prob, init_viterbi, testtagfile, seenWords):

        test_content = open(testfile)

        LM_file = open(testtagfile, "w")

        new_obs_dist = {}
        p_w = 0.01
        # assign uniform distribution for all unseen <word, tag_i>
        for pos, prob in tags.iteritems():
            new_obs_dist[pos] = math.log(((p_w * (1/float(len(seenWords)))) / (2 ** prob)), 2)

        vTags = ["VBP", "VBG", "VB", "VBN", "VBD"]   # ends with 'ed' or 'ing'
        nTags = ["NNS"]                              # ends with 's' letter
        ncTags = ["NNP", "NNS", "NN"]                # capital letter
        cdTags = ["CD"]                              # numeric

        for lNo, line in enumerate(test_content.readlines()):
            obs = nltk.WhitespaceTokenizer().tokenize(line)
            resultant_tag = []
            for ob in obs:
                # get all seen tags
                if ob not in seenWords:
                    new_viterbi = {}
                    for iVit in init_viterbi:
                        # for each tag of the test word
                        vitval = []                 # For each cell
                        vitval_index = []           # For each cell
                        for wordtag, prob in new_obs_dist.iteritems():
                            # tag = wordtag.split(" ")[1]
                            if (iVit.split(" ")[1] + " " + wordtag) in tran_prob:
                                tp = tran_prob[iVit.split(" ")[1] + " " + wordtag]
                            else:
                                tp = 0.01 * tags[wordtag]
                            if re.search(r'[A-Z][a-z]+ed|[A-Z][a-z]+ing', ob) is not None and wordtag in vTags:
                                vitval.append(init_viterbi[iVit] + tp +
                                              (new_obs_dist[wordtag] / 3))
                            elif re.search(r'[A-Z][a-z]+s', ob) is not None and wordtag in nTags:
                                vitval.append(init_viterbi[iVit] + tp +
                                              (new_obs_dist[wordtag] / 10))
                            elif re.search(r'[A-Z][a-z]+', ob) is not None and wordtag in ncTags:
                                vitval.append(init_viterbi[iVit] + tp +
                                              (new_obs_dist[wordtag] / 3))
                            elif re.search(r'\d', ob) is not None and wordtag in cdTags:
                                vitval.append(init_viterbi[iVit] + tp +
                                              (new_obs_dist[wordtag] / 10))
                            else:
                                vitval.append(init_viterbi[iVit] + tp +
                                              new_obs_dist[wordtag])
                            vitval_index.append(iVit.split(" ")[1] + " " + wordtag)
                        new_viterbi[vitval_index[vitval.index(max(vitval))]] = max(vitval)  # Get the maximum in the cell
                    resultant_tag.append(str(max(new_viterbi.iteritems(), key=operator.itemgetter(1))[0].split(" ")[1] + ">>")) # Get the maximum in the column
                    init_viterbi = new_viterbi.copy()
                else:
                    new_viterbi = {}
                    for iVit in init_viterbi:
                        # for each tag of the test word
                        vitval = []                 # For each cell
                        vitval_index = []           # For each cell
                        for wordtag, prob in obs_prob.iteritems():
                            if ob == wordtag.split(" ")[0]:
                                # get seen tag
                                tag = wordtag.split(" ")[1]
                                if (iVit.split(" ")[1] + " " + tag) in tran_prob:
                                    tp = tran_prob[iVit.split(" ")[1] + " " + tag]
                                else:
                                    tp = 0.01 * tags[tag]
                                vitval.append(init_viterbi[iVit] + tp +
                                                  obs_prob[ob + " " + tag])
                                vitval_index.append(iVit.split(" ")[1] + " " + tag)
                        new_viterbi[vitval_index[vitval.index(max(vitval))]] = max(vitval)  # Get the maximum in the cell
                    resultant_tag.append(max(new_viterbi.iteritems(), key=operator.itemgetter(1))[0].split(" ")[1]) # Get the maximum in the column
                    init_viterbi = new_viterbi.copy()

            for i in range(0, len(obs)):
                LM_file.write(obs[i] + "/" + resultant_tag[i] + " ")
            LM_file.write("\n")

        LM_file.close()
        test_content.close()

class pos_evaluation:

    @staticmethod
    def evaulate(taggedfile, reffile):

        tagged_content = open(taggedfile)
        reffile_content = open(reffile)

        totaltokens = 0
        totalKnowns = 0
        totalUnknowns = 0
        unknownCorrect = 0
        knownCorrect = 0
        delimiter = [">>"]

        for taggedline, refline in izip(tagged_content, reffile_content):
            taggedtoken = nltk.WhitespaceTokenizer().tokenize(taggedline)
            reftoken = nltk.WhitespaceTokenizer().tokenize(refline)

            totaltokens += len(taggedtoken) # get the total number of tokens

            for index, token in enumerate(taggedtoken):
                taggedtag = token.split("/")[1]
                # if unknown tag
                if ">>" in [delimit for delimit in delimiter if delimit in taggedtag]:
                    taggedtag = taggedtag.rstrip(">>")
                    totalUnknowns += 1
                    if taggedtag == reftoken[index].split("/")[1]:
                        unknownCorrect += 1
                else:
                    totalKnowns += 1
                    if taggedtag == reftoken[index].split("/")[1]:
                        knownCorrect += 1

        print "\n----------Results----------"
        print "Overall Accuracy: " + str((knownCorrect + unknownCorrect) / float(totaltokens))
        print "Known Accuracy: " + str(knownCorrect / float(totalKnowns))
        print "Unknown Accuracy: " + str(unknownCorrect / float(totalUnknowns))
        print "\n"


if __name__ == "__main__":

    print "\n-------------------------Welcome-------------------------\n"

    while True:

        option = raw_input('1. Train the Model\n2. Test the Language Model on a file\n3. Evaluate\n4. Exit\n\nEnter your choice:')

        if int(option) == 1:

            # get the training file name from the user
            trainfile = raw_input('Enter the training file:')

            # to get the lm file name from the user
            lmpath = raw_input('Enter the LM file name: ')

            # Timer to get the execution time
            start_time = time.time()

            # get the training results
            train = pos_training()
            # order: tags dictionary, transition dictionary, word_tag dictionary, no of tags in the file
            token_results = train.tokenization(trainfile)

            # get the ml of tags
            pos_prob = train.Unigram_Probability(token_results[0], token_results[3])

            print "\nFinished Calculating priors of tags........"
            # to get the transition probabilities
            tran_results = train.tagtag_probability(token_results[1], token_results[0], pos_prob)

            print "\nFinished Calculating transition probability........"
            # to get the and observation probabilities
            obs_results = train.wordtag_probability(token_results[2], token_results[0])

            print "\nFinished Calculating observation probability........"
            print "\nWriting Results to Language Model file........"
            # store the language model file
            LM_file = open(lmpath, "w")

            # to store pos_prob
            LM_file.write("pos:\n")
            for tag in pos_prob:
                LM_file.write(str(tag) + "\t" + str(math.log(pos_prob[tag], 2)) + "\n")

            # to store transition probs
            LM_file.write("transition:\n")
            for tag_tag in tran_results:
                LM_file.write(str(tag_tag) + "\t" + str(math.log(tran_results[tag_tag], 2)) + "\n")

            # to store observation probs
            LM_file.write("observation:\n")
            for word_tag in obs_results:
                LM_file.write(str(word_tag) + "\t" + str(math.log(obs_results[word_tag], 2)) + "\n")

            LM_file.close()

            print "\n--- %s seconds ---\n" % (time.time() - start_time)

        elif int(option) == 2:

            # to get the lm file name from the user
            lmfile = raw_input('Enter the LM file name: ')

            # get the test file name from the user
            testfile = raw_input('Enter the test file:')

            # get the testtag file name from the user
            testtagfile = raw_input('Enter the test tag file name:')

            # tags, tagtag_probability, wordtag_probability, init_viterbi, seenWords
            lmData = pos_testing.read_lmfile(lmfile)

            print "\nFinished Reading Model file........"
            # Timer to get the execution time
            teststart_time = time.time()

            print "\nTesting........"

            pos_testing.tag_testfile(testfile, lmData[0], lmData[1], lmData[2], lmData[3], testtagfile, lmData[4])

            print "\n--- %s seconds ---\n" % (time.time() - teststart_time)

        elif int(option) == 3:
            # to get the tagged file name from the user
            taggedfile = raw_input('Enter the tagged file name: ')

            # get the ref file name from the user
            reffile = raw_input('Enter the ref test file name:')

            pos_evaluation.evaulate(taggedfile, reffile)

        elif int(option) == 4:
            print "-------------------------Good Bye-------------------------"
            break

        else:
            print "------Your choice is not valid. Enter a valid choice!------\n"





