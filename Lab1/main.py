from skimage import data, io, filters, novice

picture = novice.open(data.coins())
width, height = picture.size()
print(width)
print(height)
edges = filters.roberts(picture)
io.imshow(edges)
io.show()