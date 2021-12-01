from PIL import Image

count = 1
for x in range(518,9948):
#for x in range(518,520):
	if x<1000 :
		number = "0"+str(x)
	else:
		number = str(x)
	try:
		image = Image.open('../876x657/heon_IMG_{}.JPG'.format(number))
		new_image = image.resize((800, 600))
		new_image.save('../800x600/{}.jpg'.format(str(count)))
		count+=1
	except: 
		continue
	

	