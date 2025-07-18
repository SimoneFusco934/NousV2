# Detect OS
ifeq ($(OS),Windows_NT)
    RM := del /Q
    SDLFLAGS := -lmingw32 -lSDL2main -lSDL2 -lSDL2_ttf
else
    RM := rm -f
    SDLFLAGS := -lSDL2 -lSDL2_ttf
endif

# Compiler and flags
CXX := g++
CXXFLAGS := -O3 -march=native -ftree-vectorize -fopenmp
LDFLAGS := $(SDLFLAGS)

# Directories
SRCDIR := ./cppFiles
HDRDIR := ./hppFiles
OBJDIR := ./objectFiles

# UseModel file
USE_SRC := UseModel.cpp
USE_OBJ := UseModel.o

# CreateModel file
CREATE_SRC := CreateModel.cpp
CREATE_OBJ := CreateModel.o

# Source files to compile (excluding UseModel/CreateModel)
SOURCES := $(filter-out $(USE_SRC) $(CREATE_SRC), $(wildcard $(SRCDIR)/*.cpp))
OBJECTS := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SOURCES))

# Default build rule
all: CreateModel UseModel

# Link final executable (create model file)
CreateModel: $(CREATE_OBJ) $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(CREATE_OBJ) $(OBJECTS) -o $@ $(LDFLAGS)

# Link final executable (main)
UseModel: $(USE_OBJ) $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(USE_OBJ) $(OBJECTS) -o $@ $(LDFLAGS)

# Compile CreateModel file
$(CREATE_OBJ): $(CREATE_SRC)
	$(CXX) -c $< -o $@

# Compile UseModel source
$(USE_OBJ): $(USE_SRC)
	$(CXX) -c $< -o $@

# Pattern rule to compile other .cpp into .o
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(HDRDIR)/%.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	$(RM) CreateModel UseModel $(USE_OBJ) $(CREATE_OBJ) $(OBJECTS)


