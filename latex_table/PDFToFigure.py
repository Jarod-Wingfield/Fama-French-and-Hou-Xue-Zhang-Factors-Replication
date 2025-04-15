# Convert pdf to figure (Remind after running the latex code)
from pdf2image import convert_from_path

images = convert_from_path("Tables.pdf", dpi=300)

images[0].save("Tables.png", "PNG")