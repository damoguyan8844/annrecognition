INSTALL_BIN_DIR = bin

install: all
	mv ui/photobook $(INSTALL_BIN_DIR)/
	mv util/annotate util/results util/convert_to_binary util/concat $(INSTALL_BIN_DIR)/
	mv feature/histo feature/rgb2ohta feature/rgb2gray $(INSTALL_BIN_DIR)/
	mv tree/nnhierarchy tree/simplify $(INSTALL_BIN_DIR)/

all: framer tpm type libphoto xtpm ui util matrix feature tree
OP = 

PHONY:
# this line only works in GNU make
#.PHONY: install all framer tpm type libphoto xtpm ui

framer: PHONY
	cd framer; $(MAKE) $(OP)
tpm: PHONY
	cd tpm; $(MAKE) $(OP)
type: PHONY
	cd type; $(MAKE) $(OP)
libphoto: PHONY
	cd libphoto; $(MAKE) $(OP)
xtpm: PHONY
	cd xtpm; $(MAKE) $(OP)
ui: PHONY
	cd ui; $(MAKE) $(OP)
util: PHONY
	cd util; $(MAKE) $(OP)
matrix: PHONY
	cd matrix; $(MAKE) $(OP)
feature: PHONY
	cd feature; $(MAKE) $(OP)
tree: PHONY
	cd tree; $(MAKE) $(OP)
