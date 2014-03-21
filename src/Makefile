all: hws

hws: main.o RadonTransform.o GenericHelpers.o GaborFilter.o Features.o ProcessingHelper.o PreProcessing.o Processing.o SWTMarshaller.o StrokeWidthTransform.o
	g++ -w -o hws main.o RadonTransform.o GenericHelpers.o GaborFilter.o Features.o ProcessingHelper.o PreProcessing.o Processing.o SWTMarshaller.o StrokeWidthTransform.o `pkg-config --libs --cflags opencv`

main.o: main.cpp
	g++ -w -c main.cpp `pkg-config --cflags opencv`

Features.o: Features.cpp
	g++ -w -c Features.cpp `pkg-config --cflags opencv`

RadonTransform.o: RadonTransform.cpp 
	g++ -w -c RadonTransform.cpp `pkg-config --cflags opencv`

GaborFilter.o: GaborFilter.cpp
	g++ -c GaborFilter.cpp `pkg-config --cflags opencv`

ProcessingHelper.o: ProcessingHelper.cpp
	g++ -c ProcessingHelper.cpp `pkg-config --cflags opencv`

PreProcessing.o: PreProcessing.cpp
	g++ -w -c PreProcessing.cpp `pkg-config --cflags opencv` 

Processing.o: Processing.cpp
	g++ -w -c Processing.cpp `pkg-config --cflags opencv`

StrokeWidthTransform.o: StrokeWidthTransform.cpp
	g++ -w -c StrokeWidthTransform.cpp `pkg-config --cflags opencv`

GenericHelpers.o: GenericHelpers.cpp
	g++ -w -c GenericHelpers.cpp `pkg-config --cflags opencv`

SWTMarshaller.o: SWTMarshaller.cpp GenericHelpers.o StrokeWidthTransform.o
	g++ -w -c SWTMarshaller.cpp StrokeWidthTransform.o `pkg-config --cflags opencv`

clean:
	rm *.o
