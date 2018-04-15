shoplist=['apple', 'mango', 'a', 'b']

print('I have', len(shoplist), 'items to purchase')

print('These items are:', end='')
for item in shoplist: 
	print(item, end=' ')

print('\nI also have to buy rice.')
shoplist.append('rice')
print('My shopping list is now', shoplist)

print('I will sort my list now')
shoplist.sort()
print('Sorted shopping list is', shoplist)

print('The first item I Will buy is', shoplist[0])
olditem = shoplist[0]
del shoplist[0]
print('I bought the', olditem)
print('My shopping list is now', shoplist)

shop=['apple','mango','carrot','banana']
name='swaroop'
