# 4 is lualatex
$pdf_mode = 4; 
$lualatex = 'lualatex -shell-escape -interaction=nonstopmode %O %S';
$biber = 'biber %O %S';
$clean_ext = 'aux bbl bcf blg fls log nav out run.xml snm synctex.gz toc';
# luamml generates files like main-luamml-mathml.html that $clean_ext can't
# match (it uses a hyphen, not a dot). Remove them explicitly.
push @generated_exts, 'luamml-mathml.html';
$clean_full_ext = '*-luamml-mathml.html';