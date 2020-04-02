# To display:
# 	make
#
# To build svg:
#
#	make covid_silhouette_a3.svg
#
# To build the covid.png bitmap
#
#	make covid.png

PYTHON = python
VPYPE = vpype -vH
PIPELINE = msimage ../module_sets/ms_thin_v1 covid_silhouette.png linemerge -t 0.1mm linesort scale -p --to 40cm 27.7cm

show: Makefile covid_silhouette.png
	$(VPYPE) $(PIPELINE) show -h

covid_silhouette_a3.svg: Makefile covid_silhouette.png
	$(VPYPE) $(PIPELINE) write -cl -p a3 covid_silhouette_a3.svg

covid.png: Makefile
	$(PYTHON) covid_to_bitmap.py
