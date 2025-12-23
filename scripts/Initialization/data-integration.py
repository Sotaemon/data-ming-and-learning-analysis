import pandas as pd

test_dict1={'id':[1,2,3,4,5,6],'name':['Alice','Bob','Cindy','Eric','Helen','Grace'],'math':[88,89,99,78,97,93],'english':[89,94,80,94,94,90]}
df1 = pd.DataFrame(test_dict1)
print(df1)

test_dict2={'id':[1,2,3,4,5,6],'name':['Alice','Bob','Cindy','Eric','Helen','Grace'],'sex':['female','male','female','female','female','female']}
df2 = pd.DataFrame(test_dict2)
print(df2)

df1.merge(df2)
print(df1)

print(pd.concat([df1,df2], axis=1))
print(pd.concat([df1,df2], axis=0))

df1._append(df2)
print(df1)