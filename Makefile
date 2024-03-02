SRC_DIR := slides_src
DEST_DIR := dest
EMBED := false

RATIO := 16_9
ifeq ($(RATIO), 16_9)
	WIDTH := 1920
	HEIGHT := 1080
	CSS_PATH := ./assets/style_ratio_169.css
else ifeq ($(RATIO), 4_3)
	WIDTH := 1024
	HEIGHT := 768
	CSS_PATH := ./assets/style_ratio_43.css
else
	WIDTH := 1920
	HEIGHT := 1080
	CSS_PATH := ./assets/style_ratio_169.css
endif

SOURCES := $(wildcard $(SRC_DIR)/*.md)

# Define object files
OBJECTS := $(patsubst $(SRC_DIR)/%.md, $(DEST_DIR)/%.html, $(SOURCES))

all: $(OBJECTS)

$(DEST_DIR)/%.html: $(SRC_DIR)/%.md
	mkdir -p $(DEST_DIR)
	pandoc --version
	pandoc $< -o $@ \
	-t revealjs \
	-s \
	--embed-resource=$(EMBED) \
	-V theme=simple \
	-V width=$(WIDTH) \
	-V height=$(HEIGHT) \
	-V controlsTutorial=false \
	-V progress=false \
	-V hash=true \
	-V center=false \
	-V slideNumber="'c/t'" \
	-V transitionSpeed='fast' \
	--include-in-header=$(CSS_PATH) \
	--katex=https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/ \

clean:
	rm -rf $(DEST_DIR)/*.html

.PHONY: all clean
