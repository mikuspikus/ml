import pandas 
data = pandas.read_csv('E:/Datasets/titanic.csv', index_col = 'PassengerId')

answers = {}

buffer = data['Sex'].value_counts()
answers[1.1] = buffer['male']
answers[1.2] = buffer['female']

answers[2] = data['Survived'].value_counts()[1]/len(data['Survived']) * 100

answers[3] = data['Pclass'].value_counts()[1]/len(data['Pclass']) * 100

answers[4.1] = data['Age'].mean()
answers[4.2] = data['Age'].median()

answers[5] = data.corr()['SibSp']['Parch']

def name_cleaning(name):
    buffer = name.replace('"', '', 2)
    buffer = buffer.replace(')', '')
    return buffer

name_data = data['Name'].get_values()
names = {}
for full_name in name_data:
    is_woman = ('Miss.' in full_name or 'Mrs.' in full_name)
    if is_woman:
        first_name = full_name[full_name.find('(') + 1 if '(' in full_name else full_name.find('.') + 2:]
        first_name = first_name[:first_name.find(' ') if ' ' in first_name else len(first_name)]
        first_name = name_cleaning(first_name)
        if first_name in names:
            names[first_name] += 1
        else:
            names[first_name] = 1
            
answers[6] = sorted(names.keys(), key = names.get, reverse = True)[0]

list_of_tasks = sorted(list(answers))
for i in list_of_tasks:
    print(i, ':', answers[i])