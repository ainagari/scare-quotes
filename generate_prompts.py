
### script to generate examples for LLMs

import re



'''
Parts of the prompt:

1. Presentation of the task: usages of quotes. Description and examples of each case (they can be made up).
2. Asking the model to output only SQ, NSQ, or Both.
3. What we show for every instance: 
a. the whole utterance, with the target usage marked with <target> </target>.
b. Same, but also showing the immediately previous utterance.

'''


introduction = "You are tasked with determining whether a usage of quotation marks constitutes an instance of scare quotes (S), non-scare quotes (N), or both (B)."
sq_description = "We consider a quoted word or phrase to be an instance of scare quotes (S) if one of the following is true:\na) The word is not to be taken at face value. Perhaps because it is a suboptimal word choice (e.g., because it doesn’t fully apply to the situation being described, or because the writer is unsure that this is the correct word to use), because it is used in a figurative sense, or with irony (e.g., \"the 'generous' lady donated one cent to charity\").\nb) The writer seems to want to distance themselves from the usage. Perhaps because the word is connotatively loaded or controversial, or because it comes from a different register (e.g., \"This lady works as an 'influencer' on Instagram\").\nc) The word itself is very unusual, incorrect, or made-up.\nd) The writer is quoting someone else’s words, but expressing, implicitly or explicitly, some attitude towards what is said (for example, that they would have said it in a different way): \"he said he was 'the humblest person'\"."
nsq_description = "The main characteristic of non-scare quotes (NS) is that the quoted passages are to be taken and interpreted at face value. There is no added attitude, no intention by the writer to warn the reader about a non-standard word or word usage or to distance themselves from what is said. The two most well-known functions of non-scare quotes are meta-linguistic mentioning (\"'Cat' has three letters\") and citing someone else's words (\"He said 'I’m hungry'\"). They can also be used to mark specialized terminology or titles (\"Have you watched 'Gone with the wind?'\"), abstract categories, or for emphasis."
instructions = "Please determine the type of quotes of the usage enclosed between <target> and </target> in the following utterance. Reply only \"S\", \"N\", or, if you perceive characteristics of both,  \"B\" (for example, a meta-linguistic comment about a word's meaning accompanied with some attitude or distancing)."
examples = "Examples:\nExample 1:\nUtterance: I do not know the meaning of <target>'diplomacy'</target>\nAnswer:N\n\nExample 2:\nUtterance: This so-called <target>'diet'</target> allows him to eat ice-cream.\nAnswer: S\n\nExample 3: Since <target>'both sides'</target> is a fundamentally inaccurate description of US politics, you have to resort to hyperbole to present a balanced picture.\nAnswer: B\n"
context = "We also include the preceding utterance for reference."

sq_description2 = "An expression in scare quotes (S) is not to be taken at face value, and the writer wishes to distance themselves from it or expresses some attitude about it, perhaps because the word is unusual, incorrect or controversial."
nsq_description2 = "An expression in non-scare quotes (N) is to be interpreted at face value. They are usually used for meta-linguistic mentioning, to cite someone else's words, or to mark specialized terminology, titles, or abstract categories."

introduction3 = "You are a classifier that takes a text with a word or phrase in quotes as input and determines whether it is an instance of scare quotes."
sq_description3 = "An expression in scare quotes is not to be taken at face value, and the writer wishes to distance themselves from it or expresses some attitude about it, perhaps because the word is unusual, incorrect or controversial."
instructions3 = "Please answer 'Yes' if the expression between <target> and </target> is an instance of scare quotes or 'No' otherwise, as in these examples:"
examples3 = "\nExample 1:\nUtterance: I do not know the meaning of <target>'diplomacy'</target>\nAnswer:No\n\nExample 2:\nUtterance: This <target>'diet'</target> allows him to eat ice-cream.\nAnswer: Yes\n" #\nExample 3: Since <target>'both sides'</target> is a fundamentally inaccurate description of US politics, you have to resort to hyperbole to present a balanced picture.\nAnswer: Both\n"
context = "We also include the preceding utterance for reference."


prompt_dict = {1:{'context':False}, 2:{'context':False}, 3:{'context':False}}


task_descriptions = dict()

task_descriptions[1] = " ".join([introduction, sq_description, nsq_description, instructions, examples])
task_descriptions[2] = " ".join([introduction, sq_description2, nsq_description2, instructions, examples])
task_descriptions[3] = " ".join([introduction, sq_description3, instructions3, examples3])


#############


def create_instance_for_prompt(dialogue):

    # Look for the target first

    # replacing code with target labels
    pattern = r"(.)<span style='color: #1924F7; font-weight: bold;'>(.*?)</span>(.)"
    new_text = re.sub(pattern, r"<target>\1\2\3</target>", dialogue)

    #pattern2 =  r"(<b>.*?</b>:.*?<target>.*?</target>.*?)(?=(<b>.*?</b>:)|$)"
    utterances = re.split(r'<b>.*?</b>:', new_text, flags=re.DOTALL)
    utterances_with_target = []
    for u in utterances:
        if '<target>' in u:
            # Optionally trim trailing tags like <p></p></div>
            end_match = re.search(r'(.*?)<p></p></div>', u, flags=re.DOTALL)
            if end_match:
                utterances_with_target.append(end_match.group(1).strip())
            else:
                utterances_with_target.append(u.strip())

    assert len(utterances_with_target) == 1
    main_utterance = utterances_with_target[0]

    return main_utterance 



def create_prompt_intro(prompt_type):
    return task_descriptions[prompt_type]



