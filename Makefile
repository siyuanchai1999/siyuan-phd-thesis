NAME := main
TEXS := $(wildcard *.tex)
TABLES := $(wildcard tables/*.tex)
NUMBERS := $(wildcard numbers/*.tex)
PLOTS := $(wildcard data/plots/*.eps)
CODE := $(wildcard code/*.java)
BIBS := $(wildcard *.bib)

%.pdf: %.fig 
	fig2dev -L eps -f Roman $*.fig >$*.eps

PYTHON313 := /opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/bin
LUALATEX := PATH="$(PYTHON313):$$PATH" lualatex -shell-escape -interaction=nonstopmode

all: ${NAME}.pdf

${NAME}.pdf: ${TEXS} ${TABLES} ${NUMBERS} ${BIBS} ${CODE} ${PLOTS}
	$(LUALATEX) $(NAME)
	biber $(NAME)
	$(LUALATEX) $(NAME)
	$(LUALATEX) $(NAME)
	@echo '****************************************************************'
	# @dvips -t letter -o $(NAME).ps $(NAME).dvi
	# @ps2pdf -dPDFSETTINGS=/prepress $(NAME).ps $(NAME).pdf
	@echo '******** Did you spell-check the paper? ********'

clean:
	ls $(NAME)* | grep -v ".tex" | grep -v ".bib" | xargs rm -f
	rm -f *.bak *~