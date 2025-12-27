import random

def splitClauses(text):
    return map(lambda x: x.strip(' '),
        text.lower().replace( ',', '.' ).replace( "'", '' ).split('.'))

def splitWords(text):
    return text.split( ' ' )

conversions = { 'i' : 'you',  'you' : 'i',
                'am' : 'are', 'are' : 'am',
                'my' : 'your', 'your' : 'my',
                'myself' : 'yourself', 'yourself' : 'myself'}
rules = { "you": (1,
 [
  [[0, "you", ["want", "need"], 0],
    ["what would it mean to you if you got", 4],
    ["why do you want", 4],
    ["suppose you got", 4, "soon"],
    ["what if you never got", 4],
    ["what would getting", 4, "mean to you"]],
  [[0,"you","are",0,["unhappy", "sick"], 0],
    ["I'm sorry to hear you are", 5],
    ["do you think coming here will help you not be", 5],
    ["I'm sure it's not pleasant to be", 5],
    ["can you explain what made you", 5]],
  [[0,"you","are", 0,["happy","glad","better"], 0],
    ["how have I helped you to be", 5],
    ["has your treatment made you", 5],
    ["what makes you", 5, "just now"],
    ["can you explain why you are suddenly", 5]],
  [[0,"you",["feel","think","wish"],"you", 0],
    ["do you really think so"],
    ["but you are not sure you", 5],
    ["do you really doubt you", 5]],
  [[0, "you", "are", 0],
    ["is it because you are",4,"that you came to me"],
    ["how long have you been", 4],
    ["do you believe it is normal to be", 4],
    ["do you enjoy being", 4]],
  [[0, "you", ["cant", "cannot"], 0],
    ["how do you know you can't", 4],
    ["have you tried"],
    ["perhaps you could", 4, "now"],
    ["do you really want to be able to", 4]],
  [[0, "you", "don't", 0],
    ["don't you really", 4],
    ["why don't you", 4],
    ["do you wish to be able to", 4],
    ["does that trouble you"]],
  [[0, "you", "feel", 0],
    ["tell me more about such feelings"],
    ["do you often feel", 4],
    ["do you enjoy feeling", 4],
    ["of what does feeling", 4, "remind you"]],
  [[0, "you", 0, "i", 0],
    ["perhaps in your fantasies we", 3, "each other"],
    ["do you wish to", 3, "me"],
    ["you seem to need to", 3, "me"],
    ["do you", 3, "anyone else"]],
  [[0],
    ["you say", 1],
    ["can you elaborate on that"],
    ["do you say", 1, "for special reasons"],
    ["that's quite interesting"]]
 ] ),
 "i": (0,
 [
  [[0, "i", "am", 0],
    ["what makes you think i am", 4],
    ["does it please you to believe i am", 4],
    ["do you sometimes wish you were", 4],
    ["perhaps you would like to be", 4]],
  [[0, "i", 0, "you"],
    ["why do you think i", 3, "you"],
    ["you like to think i", 3, "you, don't you"],
    ["what makes you think i", 3, "you"],
    ["really, i", 3, "you"],
    ["do you wish to believe that i", 3, "you"],
    ["suppose i did",3,"you, what would that mean"],
    ["does someone else believe i", 3, "you"]],
  [[0, "i", 0],
    ["we were discussing you, not me"],
    ["you're not really talking about me, are you"],
    ["what are your feelings now"]]
 ] ),
 "was": (2,
 [
  [[0, "was", "you", 0],
    ["what if you were", 4],
    ["do you think you were", 4],
    ["were you", 4],
    ["what would it mean if you were", 4],
    ["what does '", 4, "' suggest to you"]],
  [[0, "you", "was", 0],
    ["were you really"],
    ["why do you tell me you were", 4, "now"],
    ["perhaps I already knew you were", 4]],
  [[0, "was", "i", 0],
    ["would you like to believe I was", 4],
    ["what suggests that I was", 4],
    ["what do you think"],
    ["perhaps I was", 4],
    ["what if I had been", 4]]
 ] )
}

from ELIZArules import conversions, rules

def transform(text):
    return map(lambda x: \
              conversions[x] if x in conversions else x,
          text)

def matchPattern(pattern, text):
    matches = []
    for i in range(len(pattern)):
        if pattern[i] == 0:
            if i + 1 == len(pattern):
                matches.append(' '.join(text))
            else:
                success, match, text = \
                    matchWildcard(pattern[i+1], text)
                if not success:
                    return False, []
                else:
                    matches.append(match.strip())
                    
        elif len(text) == 0:
            return False, []
            
        elif not isinstance(pattern[i], str):
            if text[0] in pattern[i]:
                matches.append(text[0])
                text = text[1:]
            else:
                return False, []
                
        elif pattern[i] == text[0]:
            matches.append(text[0])
            text = text[1:]
            
        else:
            return False, []
    return True, matches

def matchWildcard(term, text, soFar=''):
    if len(text) == 0:
        return False, '', []

    if (isinstance(term, str) and text[0] == term) or\
            (not isinstance(term,str) and text[0] in term):
        return True, soFar, text

    return matchWildcard(term,
                         text[1:],
                         soFar + ' ' + text[0])

def findKeywords(text):
    for phrase in splitClauses(text):
        words = list(transform(splitWords(phrase)))
        maxP = 0
        keywords = [ ]
        for word in words:
            
            if word in rules:
                p = rules[word][0]
                if p > maxP:
                    keywords.insert(0,word)
                    maxP = p
                else:
                    keywords.append(word)
                    
        if len(keywords) > 0:
            return keywords, words
    return [ ],[ ]
    
def answer(keywords, words):
    for keyword in keywords:
        for test in rules[keyword][1]:

            pattern = test[0]
            responses = test[1:]

            success, matches = matchPattern(pattern, words)
            if success:
                response = random.choice(responses)
                return compose(response, matches)

def compose(template, fields):
    result = ""
    for t in template:
        if type(t) is int:
            result += " " + fields[t-1]
        else:
            result += " " + t
    return result.strip()

def testPatterns():
    print(matchWildcard('stop', ['stop']))
    print(matchWildcard('stop', ['stop', 'now']))
    print(matchWildcard('stop', ['I', 'stop', 'now']))
    print(matchWildcard('stop', ['I', 'can', 'stop']))
    print(matchWildcard('stop',['I','can','stop','now']))
    print(matchWildcard('stop', ['I', 'can', 'finish', 'now']))
    print(matchPattern([0,'stop',0], ['stop']))
    print(matchPattern([0,'stop',0], ['stop', 'now']))
    print(matchPattern([0,'stop',0], ['I','stop', 'now']))
    print(matchPattern([0,'stop',0], ['I', 'can', 'stop']))
    print(matchPattern([0,'stop',0], ['I','stop', 'right','now']))
    print(matchPattern([0,'stop',0], ['I', 'can', 'finish','now']))
    
while(True):
    text = input("? ")
    if len(text) == 0:
        break

    keywords, words = findKeywords(text)
    print(answer(keywords, words))

