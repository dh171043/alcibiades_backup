# varargs
def some_function(a,b,*data):
    pass

some_function(1,2,3,4,"hello", "goodbye")


#collection unpacking
days =[ 'mon', 'tues', 'wed']
mon, tues, wed = days

# splat operation
def something(a,b,c,d,e,f=9): 
# now f is optional becuase if f is not called then it defaults to 9
    print(a, b, c, d, e, f)

data = [1,2,3,4,5,6]
something(*data)

def write_email(sender ="me", recepient ="you", message ="text"):
    if sender:
        to = f'{sender}'
    if recepient:
        sent_from  = f'{recepient}'
    if message:
        body = f'{message}'

def input_email():
    sender = input("Please enter your email")
    return sender




