import tkinter
import pickle
import  tkinter as tk
from tkinter import ttk, messagebox, simpledialog, IntVar

datafile = 'Doolittle data.dat'

class KnowledgeBase:
    def __init__(self):
        self.animals= set( )
        self.queries = [ ]
        
class Query:
    def __init__(self, question, yeses=set(), nos=set()):
        self.question = question
        self.yeses = yeses
        self.nos = nos

    def __str__(self):
        return '[Q: ' + self.question + ' Y: ' + \
                    str(self.yeses) + ' N: ' + str(self.nos) + ']'

    def ask(self, possible):
        answer = tk.messagebox.askquestion(
                                              title= 'Question', 
                                              message=self.question)
        if answer == 'yes':
            return possible.intersection(self.yeses), 'yes'
        else:
            return possible.intersection(self.nos), 'no'

def canWeFind(newAnimal):
    possible = knowledgeBase.animals
    for query in knowledgeBase.queries:

        if newAnimal in query.yeses:
            possible = possible.intersection(query.yeses)

        if newAnimal in query.nos:
            possible = possible.intersection(query.nos)
    return len(possible) == 1
    
class myCheckbutton(tk.Checkbutton):
    def __init__(self, top, variable, text,
                        row, column, option):
        self.var = tk.IntVar()
        self.variable = variable
        self.option = option
        super().__init__(top, text=text, variable=self.var,
                onvalue=1, offvalue=0,
                command=self.callback)
        self.grid(row=row, column=column)

    def callback(self):
        if self.var.get() == 1:
            self.variable.add(self.option)
        else:
            self.variable.discard(self.option)

class CompleteQueryDialog:
    def __init__(self, parent, question, options):
        self.options = options
        self.question = question
        self.yeses = set()
        self.nos = set()
        self.top  = self.window = tk.Toplevel(parent)
        self.frame = ttk.Frame(self.top, padding=10)
        self.frame.grid()

        text = 'Please check the box for each line where the answer is yes.' 
        tk.Label(self.frame, text=text).\
                grid(row=0, column=0,
                    columnspan=2, padx=10)
        self.okButton = tk.Button(self.frame, text='OK',
                    command=self.onOK)
        self.okButton.grid(row=0, column=2)

        tk.Label(self.frame, text=question).grid(row=1,
                    column=0, columnspan=2)
        tk.Label(self.frame, text='Yes').\
                   grid(row=2, column = 1)
                   
        index = 0
        widgets = [ ]
        for option in options:
            tk.Label(self.frame, text=option). \
                        grid(row=index + 3, column = 0)
            widgets.append(myCheckbutton(self.frame,
                            self.yeses, '', index + 3, 1, option))
            index = index + 1

    def onOK(self):
        self.top.destroy()


def completeQuery(question, animals, newAnimal):
    animals.add(newAnimal)
    dialog = CompleteQueryDialog(root, question, animals)
    root.wait_window(dialog.top)
    return dialog.yeses, animals - dialog.yeses

def fillInOldQueries(queries, newAnimal):
    options = [q.question for q in queries]
    question = 'Regarding a ' + newAnimal
    dialog = CompleteQueryDialog(root, question, options)
    root.wait_window(dialog.top)
    for i in range(len(options)):
        if options[i] in dialog.yeses:
            queries[i].yeses.add(newAnimal)
        else:
            queries[i].nos.add(newAnimal)
            
def giveUp(asked):

    newAnimal = ''
    while len(newAnimal) == 0:

        newAnimal = tk.simpledialog.askstring('I give up',
                            'What animal are you thinking of?')

    for q in asked:
        a = asked[q]

        if a == 'yes':
            q.yeses.add(newAnimal)
        else:
            q.nos.add(newAnimal)

    return newAnimal

def newQuery(newAnimal, wrongAnimal):
    global knowledgeBase

    question = tk.simpledialog.askstring('I give up',
           'What is a question that would differentiate a ' \
          + newAnimal + ' from a ' + wrongAnimal + '?')

    query = Query(question)

    query.yeses, query.nos = completeQuery(question,
                            knowledgeBase.animals, newAnimal)

    return query

def getBestQuery(possible, queries):
    best = None
    bestBalance = 9999999 # A very large number

    for query in queries:
        yes = query.yeses.intersection(possible)
        no =  query.nos.intersection(possible)
        balance = abs(len(yes) - len(no))

        if balance < bestBalance:
            best = query
            bestBalance = balance

    return best


def guessAnimal():
    global goButton,knowledgeBase

    goButton['state'] = tk.DISABLED
    possible= set(knowledgeBase.animals)
    notasked = list(knowledgeBase.queries)
    asked = dict()

    while True:

        if len(possible) == 0:
            knowledgeBase.animals.add(giveUp(asked))
            break

        elif len(possible) == 1:
            [guess] = possible #unpack a singleton from a set
            gotIt = tk.messagebox.askquestion(
                                    'I think I know',
                                    'Is your animal a ' + guess + '?')

            if gotIt == 'yes':
                tk.messagebox.showinfo(
                                            'Yes!', 'I am the greatest')
                print('Answered correctly because a', guess)
                for q in asked:
                    a = asked[q]
                    print('   ', q.question, '==', a)
                print('QED')

            else:
                newAnimal = giveUp(asked)
                if len(notasked) > 0:
                    fillInOldQueries(notasked, newAnimal)

                knowledgeBase.animals.add(newAnimal)

                if not canWeFind(newAnimal):
                    query = newQuery(newAnimal,guess)
                    knowledgeBase.queries.append(query)

            break

        query = getBestQuery(possible, notasked)
        if query == None:
            print('Run out of queries. asked=', asked,
                    'notasked=', notasked)
            exit(1)

        possible, response = query.ask(possible)
        asked[(query)] = response
        notasked.remove(query)

    goButton['state'] = tk.NORMAL

# Load the knowledge base
try:
    with open(datafile, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        knowledgeBase = unpickler.load( )
        print ('Loaded existing knowledge base')
except FileNotFoundError:
    knowledgeBase = KnowledgeBase( )
    print('No knowledge base found -- creating new one')

# Create the main window
root = tk.Tk( )
frame = ttk.Frame(root, padding=10)
frame.grid( )
tk.Label(frame, text='Think of an animal').\
            grid(column=0, row=0)
goButton = tk.Button(frame, text='OK',
            command=guessAnimal)
goButton.grid(column=1, row=0)

# Do the main GUI loop
root.mainloop( )



# Save the database
with open(datafile, 'wb') as f:
    pickler = pickle.Pickler(f)
    pickler.dump(knowledgeBase)
    print('Saved knowledge base')