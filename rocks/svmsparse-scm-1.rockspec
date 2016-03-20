package = "svmsparse"
version = "scm-1"

source = {
   url = "git://github.com/DavidGrangier/sparsesvm.git"
}

description = {
   summary = "Fast Sparse Linear SVM",
   detailed = [[Torch interface to Leon Bottou's fast svm sgd package
   ]],
   homepage = "https://github.com/DavidGrangier/sparsesvm/",
   license = "LGPL-2.1"
}

dependencies = {
   "torch >= 7.0",
}

build = {
   type = "make",
   install_variables = {
      INST_PREFIX="$(PREFIX)",
      INST_BINDIR="$(BINDIR)",
      INST_LIBDIR="$(LIBDIR)",
      INST_LUADIR="$(LUADIR)",
      INST_CONFDIR="$(CONFDIR)",
   },
}
