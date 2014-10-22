def flatten(content):
   return " ".join([line.split()[0] for line in content.strip().split("\n")])

for inst in open("./gene.train", "r").read().strip().split("\n\n"):
    print flatten(inst)

