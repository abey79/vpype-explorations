# Preview: make
# Show (runs optimizations): make show
# Build SVG (a4, a3, a4 landscape, a3 landscape): make any_file_name_a4.svg (_a3.svg, _a4l.svg, _a3l.svg)

PYTHON = python
VPYPE = vpype -vH
PIPELINE_VPY = pipeline.vpy
OPTIMIZE_VPY = optimize.vpy



preview: Makefile $(PIPELINE_VPY)
	$(VPYPE) -I $(PIPELINE_VPY) show -h

show: Makefile $(PIPELINE_VPY) $(OPTIMIZE_VPY)
	$(VPYPE) -I $(PIPELINE_VPY) -I $(OPTIMIZE_VPY) show -h

%_a4.svg: Makefile $(PIPELINE_VPY) $(OPTIMIZE_VPY)
	$(VPYPE) -I $(PIPELINE_VPY) -I $(OPTIMIZE_VPY) write -c -p a4 $@

%_a3.svg: Makefile $(PIPELINE_VPY) $(OPTIMIZE_VPY)
	$(VPYPE) -I $(PIPELINE_VPY) -I $(OPTIMIZE_VPY) write -c -p a3 $@

%_a4l.svg: Makefile $(PIPELINE_VPY) $(OPTIMIZE_VPY)
	$(VPYPE) -I $(PIPELINE_VPY) -I $(OPTIMIZE_VPY) write -cl -p a4 $@

%_a3l.svg: Makefile $(PIPELINE_VPY) $(OPTIMIZE_VPY)
	$(VPYPE) -I $(PIPELINE_VPY) -I $(OPTIMIZE_VPY) write -cl -p a3 $@
