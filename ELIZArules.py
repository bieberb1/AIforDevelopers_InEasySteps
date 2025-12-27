# This script is derived from the DOCTOR script for the original ELIZA program
# which was writen by Joseph Weizenbaum and published in
#
#  Weizenbaum, J. (1966). ELIZA: A computer program for the study of natural
#  language communication between man and machine.
#  Communications of the ACM, 9, 36-45.
#
# In 2021, Jeff Shrager (http://shrager.org) re-discovered Wiezenbaum's orginal
# ELIZA in the Weizenbaum archives at MIT. He has said()
# "I contacted Dr. Weizenbaum's estate for permission to open-source this code,
# and they granted this permission under a Creative Commons CC0 public domain
# license." - https://sites.google.com/view/elizagen-org/the-original-eliza
# (retrieved 15 Nov 20223)

# All other creative content in this file is Copyright (c) In Easy Steps, 2023

# conversions is a dictionary of words to replace. This changes a clause
# so that it can be repeated back to the user. When given "i am nice" we
# can produce something like "what makes you think you are nice"
# We also take the chance to conflate some exact synonyms so both will trigger
# the same rule.
conversions = {	"i"  : "you",  "you"  : "i",
				"am" : "are",  "are"  : "am",
				"my" : "your", "your" : "my",
				"me" : "you",
				"myself": "yourself", "yourself": "myself",
				"dreamed": "dreampt",
				"maybe": "perhaps"
				}

# Family is a macro of words to insert into rules. It lists all of the family members we might respond to.
Family = ["mother", "mom", "father", "dad", "sister", "brother", "wife", "husband", "children", "child", "son", "daughter"]

# AllOrNone is a macro of words meaning everybody or nobody
AllOrNone = ["everyone", "everybody", "nobody", "noone"]

# AllOrNoneRules is a macro containing a full set of rules that we need to repeat several times
AllOrNoneRules = [
				[[0, AllOrNone, 0],
					["really", 2],
					["surely not", 2],
					["can you think of anyone in particular"],
					["who, for example"],
					["you are thinking of a very sepecial person"],
					["who, may i ask"],
					["someone special, perhaps"],
					["you have a particular person in mind, don't you"],
					["who do you think you are talking about"]]
			]

# rules is a dictionary keyed on keywords. It contains tuples of the priority
# of the keyword and a list of lists.
# the first element of each list is a pattern to match and the other elements
# are patterns of possible responses.
# Those patterns are themselves lists.
rules = { "sorry": (0,
			[
				[[0],
					["please do not apologize"],
					["apologies are not necessary"],
					["what feelings do you have when you apologise"]]					
			]),
		"remember": (5,
			[
				[[0, "you", "remember", 0],
					["do you think often of", 4],
					["does thinking of", 4, "bring anything else to mind"],
					["what else do you remember"],
					["why do you remember", 4, "just now"],
					["what in the present situation reminds you of", 4],
					["what is the connection between me and", 4]],
				[[0, "do", "i", "remember", 0],
					["did you think I would forget", 5],
					["why do you think I should recall", 5, "now"],
					["what about", 5]]
			]),
		"if": (3,
			[
				[[0, "if", 0],
					["do you think it's likely that", 3],
					["do you wish that", 3],
					["what do you think about", 3],
					["really,", 2, 3]]
			]),
		"dreampt": (4,
			[
				[[0, "you", "dreampt", 0],
					["really,", 4],
					["have you ever fantasized", 4, "while you were awake"],
					["have you dreampt", 4, "before"]]
			]),
		"dream": (3,
			[
				[[0],
					["what does that dream suggest to you"],
					["do you dream often"],
					["what persons appear in your dreams"],
					["don't you belive that dream has something to do with your problem"]]
			]),
		"perhaps": (0,
			[
				[[0],
					["you don't seem quite certain"],
					["why the uncertain tone"],
					["can't you be more positive"],
					["you aren't sure"],
					["don't you know"]]
			]),
		"name": (15,
			[
				[[0],
					["I am not interested in names"]]
			]),
		"hello": (0,
			[
				[[0],
					["how do you do. Please state your problem"]]
			]),
		"computer": (50,
			[
				[[0],
					["do computers worry you"],
					["why do you mention computers"],
					["what do you think machines have to do with your problem"],
					["don't you think computers can help people"],
					["what about machines worries you"],
					["what do you think about machines"]]
			]),
		"are": (0,
			[
				[[0, "are", "you", 0],
					["do you believe you are", 4],
					["would you want to be", 4],
					["I wish I could tell you you are", 4],
					["what would it mean if you were", 4]],
				[[0],
					["why do you say 'am'"],
					["I don't understand that"]]
			]),
		"am": (0,
			[
				[[0, "are", "i", 0],
					["why are you interested in whether I am", 4, "or not"],
					["would you prefer if I weren't", 4],
					["perhaps I am", 4, "in your fantasies"],
					["do you sometimes think I am", 4]],
				[[0, "are", 0],
					["did you think they might not be", 3],
					["would you like it if they were not", 3],
					["what if they were not", 3],
					["possibly they are", 3]]
			]),
		"my": (0,
			[
				[[0, "my", 0],
					["why are you concerned over my", 3],
					["what about your own", 3],
					["are you worried about someone elses", 3],
					["really, my", 3]]
			]),
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
			]),
		"you": (1,
			[
				[[0, "you", ["want", "need"], 0],
					["what would it mean to you if you got", 4],
					["why do you want", 4],
					["suppose you got", 4, "soon"],
					["what if you never got", 4],
					["what would getting", 4, "mean to you"],
					["what does wanting", 4, "have to do with this discussion"]],
				[[0, "you", "are", 0, ["sad", "unhappy", "depressed", "sick"], 0],
					["I'm sorry to hear you are", 5],
					["do you think coming here will help you not be", 5],
					["I'm sure it's not pleasant to be", 5],
					["can you explain what made you", 5]],
				[[0, "you", "are", 0, ["happy", "elated", "glad", "better"], 0],
					["how have I helped you to be", 5],
					["has your treatment made you", 5],
					["what makes you", 5, "just now"],
					["can you explain why you are suddenly", 5]],
				[[0, "you", ["feel", "think", "believe", "wish"], "you", 0],
					["do you really think so"],
					["but you are not sure you", 5],
					["do you really doubt you", 5]],
				[[0, "you", "are", 0],
					["is it because you are", 4, "that you came to me"],
					["how long have you been", 4],
					["do you believe it is normal to be", 4],
					["do you enjoy being", 4]],
				[[0, "you", ["cant", "cannot"], 0],
					["how do you know you can't", 4],
					["have you tried"],
					["perhaps you could", 4, "now"],
					["do you really want to be able to", 4]],
				[[0, "you", "dont", 0],
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
					["do you", 3, "anyone else"]]
					#,
#				[[0],
#					["you say", 1],
#					["can you elaborate on that"],
#					["do you say", 1, "for special reasons"],
#					["that's quite interesting"]]
			]),
		"i": (0,
			[
				[[0, "i", "am", 0],
					["what makes you think i am", 4],
					["does it please you to believe i am", 4],
					["do you sometimes wish you were", 4],
					["perhaps you woud like to be", 4]],
				[[0, "i", 0, "you"],
					["why do you think i", 3, "you"],
					["you like to think i", 3, "you, don't you"],
					["what makes you think i", 3, "you"],
					["really, i", 3, "you"],
					["do you wish to belive that i", 3, "you"],
					["suppose i did", 3, "you, what would that mean"],
					["does someone else believe i", 3, "you"]],
				[[0, "i", 0],
					["we were discussing you, not me"],
					["oh, i", 3],
					["you're not really talking about me, are you"],
					["what are your feelings now"]]
			]),
		"yes": (0,
			[
				[[0],
					["you seem quite positive"],
					["are you sure"],
					["i see"],
					["i understand"]]
			]),
		"no": (0,
			[
				[[0],
					["are you saying 'no' just to be negative"],
					["are you being a bit negative"],
					["why not"],
					["why 'no'"]]
			]),
		"your": (2,
			[
				[[0, "your", 0, Family, 0],
					["tell me more about your family"],
					["who else in your family", 5],
					["your", 4],
					["what else comes to mind when you think of your", 4]],
				[[0, "your", 0],
					["your", 3],
					["why do you say your", 3],
					["does that suggest anything else that belongs to you"],
					["is it important to you that", 2, 3]]
			]),
		"can": (0,
			[
				[[0, "can", "i", 0],
					["you belive i can", 4, "don't you"],
					["you want me to be able to", 4],
					["you want to be able to", 4],
					["perhaps you would like to be able to", 4, "yourself"]],
				[[0, "can", "you", 0],
					["whether on not you can", 4, "depends on you more than on me"],
					["do you want to be able to", 4],
					["perhaps you don't want to", 4]]
			]),
		"what": (0,
			[
				[[0],
					["why do you ask"],
					["does that question interest you"],
					["what is it you really want to know"],
					["are such questions much on your mind"],
					["what answer would please you most"],
					["what do you think"],
					["what comes to mind when you ask that"],
					["have you asked such questions before"],
					["have you asked anyone else"]]
			]),
		"because": (0,
			[
				[[0],
					["is that the real reason"],
					["don't any other reasons come to mind"],
					["does that reason seem to explain anything else"],
					["what other reasons might there be"]]
			]),
		"why": (0,
			[
				[[0, "why", "dont", "i", 0],
					["do you believe i don't", 5],
					["perhaps i will", 5, "in good time"],
					["should you", 5, "yourself"],
					["you want me to", 5]],
				[[0, "why", "cant", "you", 0],
					["do you think you should be able to", 5],
					["do you want to be able to", 5],
					["do you really think this will help you to", 5],
					["have you any idea why you can't", 5]]
			]),
		"everyone": (2, AllOrNoneRules),
		"everyone": (2, AllOrNoneRules),
		"everybody": (2, AllOrNoneRules),
		"nobody": (2, AllOrNoneRules),
		"noone": (2, AllOrNoneRules),
		"always": (1,
			[
				[[0],
					["can you think of a specific example"],
					["when"],
					["what incident are you thinking of"],
					["really, always"]]
			]),
		"like": (10,
			[
				[[0, ["am", "is", "are", "was"], 0, "like", 0],
					["in what way"],
					["what resemblance so you see"],
					["what does that similarity suggest to you"],
					["what other connections do you see"],
					["what do you suppose that resemblance means"],
					["what is the connection, do you suppose"],
					["could there be some connection"],
					["how"]]
			])		
		}

