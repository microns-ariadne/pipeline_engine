'''Pickle a classifier - create a .pkl file for the Luigi pipeline

'''

import Tkinter
import tkMessageBox
from tkFileDialog import askopenfilename, asksaveasfile, askopenfile
import tkFont
import ttk
import os
import re
import cPickle
import webbrowser
from ariadne_microns_pipeline.classifiers import all_classifiers
from ariadne_microns_pipeline.algorithms.normalize import NormalizeMethod

help_url = "https://github.com/microns-ariadne/pipeline_engine/blob/" \
    "use_luigi/ariadne_microns_pipeline/scripts/pickle_a_classifier.md"

class XYZWidget(object):
    '''A widget with x, y and z values'''
    def __init__(self, master, name):
        '''Initialize the widget
        
        :param master: the parent widget that acts as a panel for the x, y, z
        children
        '''
        self.pane = Tkinter.PanedWindow(master)
        self.name = name
        self.lx = Tkinter.Label(self.pane, text="x")
        self.lx.pack(side="left")
        self.wx = Tkinter.Entry(self.pane)
        self.wx.configure(dict(width=4))
        self.wx.pack(side="left")
        self.ly = Tkinter.Label(self.pane, text="y")
        self.ly.pack(side="left")
        self.wy = Tkinter.Entry(self.pane)
        self.wy.configure(dict(width=4))
        self.wy.pack(side="left")
        self.lz = Tkinter.Label(self.pane, text="z")
        self.lz.pack(side="left")
        self.wz = Tkinter.Entry(self.pane)
        self.wz.configure(dict(width=4))
        self.wz.pack(side="left")
    
    def grid(self, *args, **kwargs):
        self.pane.grid(*args, **kwargs)
        
    def validate(self):
        '''Return True if the widget has valid values'''
        for widget, name in (self.wx, "x"), (self.wy, "y"), (self.wz, "z"):
            if re.match("^\\d+$", widget.get()) is None:
                widget.focus()
                tkMessageBox.showerror(
                    "%s: %s is not a number", (self.name, name))
                return False
        return True
    
    @staticmethod
    def __set(widget, value):
        widget.delete(0)
        widget.insert(0, str(value))
    
    @staticmethod
    def __get(widget):
        return int(widget.get())
    
    def get_x(self):
        return XYZWidget.__get(self.wx)
    
    def set_x(self, value):
        XYZWidget.__set(self.wx, value)
    
    x = property(get_x, set_x)
        
    def get_y(self):
        return XYZWidget.__get(self.wy)
    
    def set_y(self, value):
        XYZWidget.__set(self.wy, value)
    
    y = property(get_y, set_y)
    
    def get_z(self):
        return XYZWidget.__get(self.wz)
    
    def set_z(self, value):
        XYZWidget.__set(self.wz, value)
    
    z = property(get_z, set_z)
    
def main():
    root = Tkinter.Tk()
    tkFont.nametofont("TkDefaultFont").configure(family='sans-serif')
    root.title("Pickle a classifier")
    lfchoice = Tkinter.LabelFrame(root, text="Classifier")
    lfchoice.pack(fill="both", expand="yes", padx=10, pady=5, side="top")
    pane = Tkinter.PanedWindow(root)
    pane.pack(fill="both", expand="yes", padx=10, pady=5, side="bottom")
    choice = ttk.Combobox(lfchoice, 
                          values = sorted(all_classifiers.keys()),
                          state="readonly")
    choice.set(sorted(all_classifiers.keys())[0])
    choice.pack(side="left")
    gochoice = Tkinter.Button(
        lfchoice, text="Go", command=lambda : go_classifier(choice, pane))
    gochoice.pack(side="right")
    
    help_button = Tkinter.Button(
        lfchoice, text="Help", command=lambda: webbrowser.open(help_url))
    help_button.pack(side="right")

    root.mainloop()

def go_classifier(classifier_choice, pane):
    classifier_name = classifier_choice.get()
    top_level = classifier_choice.master.master
    if classifier_name == "keras":
        go_keras(pane)
    elif classifier_name == "aggregate":
        go_aggregate(pane)
    elif classifier_name == "caffe":
        go_caffe(pane)

def ask_filename(entry, filetypes):
    result = askopenfilename(filetypes = filetypes)
    if result != "":
        entry.delete(0)
        entry.insert(0, result)

def clear_pane(pane):
    '''Clear all of the children in a pane'''
    children = pane.children.values()
    for child in children:
        child.destroy()

def go_keras(pane):
    clear_pane(pane)
    l1 = Tkinter.Label(pane, text="Model file")
    l1.grid(row=0, column=0)
    t1 = Tkinter.Entry(pane)
    t1.configure(dict(width=40))
    t1.grid(row=0, column=1)
    b1 = Tkinter.Button(pane, text="Select",
                        command=lambda:ask_filename(
                            t1, [("Model files", ".json"),
                                 ("All files", ".*")]))
    b1.grid(row=0, column=2,)
    l2 = Tkinter.Label(pane, text="Weights file")
    l2.grid(row=1, column=0)
    t2 = Tkinter.Entry(pane)
    t2.configure(dict(width=40))
    t2.grid(row=1, column=1)
    b2 = Tkinter.Button(pane, text="Select",
                        command=lambda:ask_filename(
                            t2, [("Weights files", ".h5"),
                                 ("All files", ".*")]))
    b2.grid(row=1, column=2)
    
    l3 = Tkinter.Label(pane, text="Input size")
    l3.grid(row=2, column=0)
    
    iw = XYZWidget(pane, "Input size")
    iw.grid(row=2, column=1, sticky="w")
    
    l4 = Tkinter.Label(pane, text="Output size")
    l4.grid(row=3, column=0)
    
    ow = XYZWidget(pane, "Output size")
    ow.grid(row=3, column=1, sticky="w")
    
    l5 = Tkinter.Label(pane, text="Cropping size")
    l5.grid(row=4, column=0)
    
    cw = XYZWidget(pane, "Cropping size")
    cw.x = 0
    cw.y = 0
    cw.z = 0
    cw.grid(row=4, column=1, sticky="w")
    
    lcn = Tkinter.Label(pane, text="Class names")
    lcn.grid(row=5, column=0)
    class_name_widget = Tkinter.Entry(pane)
    class_name_widget.configure(dict(width=40))
    class_name_widget.grid(row=5, column=1)
    ln = Tkinter.Label(pane, text="Normalization")
    ln.grid(row=6, column=0)
    normalize_kwds = [_.name for _ in NormalizeMethod]
    normalization_widget = ttk.Combobox(
        pane,
        values = normalize_kwds,
        state="readonly")
    normalization_widget.grid(row=6, column=1)
    normalization_widget.set(NormalizeMethod.NONE.name)
    
    def make_classifier():
        from tkMessageBox import showerror
        model_file = t1.get()
        if not os.path.exists(model_file):
            showerror("No such model file", "%s does not exist" % model_file)
            return
        weights_file = t2.get()
        if not os.path.exists(weights_file):
            showerror("No such weights file", "%s does not exist" % weights_file)
        if not iw.validate():
            return
        if not ow.validate():
            return
        if not cw.validate():
            return
        keras = all_classifiers["keras"]
        xypad_size = iw.x - ow.x
        zpad_size = iw.z - ow.z
        xy_trim_size = cw.x
        z_trim_size = cw.z
        normalization = NormalizeMethod[normalization_widget.get()]
        class_names = [_.strip() for _ in class_name_widget.get().split(",")]
        classifier = keras(model_path=model_file,
                           weights_path=weights_file,
                           xypad_size=xypad_size,
                           zpad_size=zpad_size,
                           block_size=(iw.z, iw.y, iw.x),
                           normalize_method=normalization,
                           xy_trim_size=xy_trim_size,
                           z_trim_size=z_trim_size,
                           classes = class_names)
        return classifier
    
    def save():
        classifier = make_classifier()
        with asksaveasfile(filetypes=[("Pickle file", ".pkl"),
                                      ("All files", ".*")]) as fd:
            cPickle.dump(classifier, fd)
            
    save = Tkinter.Button(pane, text="Save", command=save)
    save.grid(row=7, column=2)

def go_aggregate(pane):
    clear_pane(pane)
    ladd = Tkinter.Label(pane, text="Add a classifier")
    ladd.grid(row=0, column=0)
    classifier_paths = []
    class_names = []
    def save():
        aggregate = all_classifiers["aggregate"]
        name_maps = [dict([(k, v.get()) for k, v in d.items()])
                     for d in class_names]
        classifier = aggregate(classifier_paths, name_maps)
        with asksaveasfile(filetypes=[("Pickle file", ".pkl"),
                                          ("All files", ".*")]) as fd:
            cPickle.dump(classifier, fd)
    bsave = Tkinter.Button(pane, text="Save", command=save)
    bsave.grid(row=1, column=2)
    def add_a_classifier():
        with askopenfile(
            filetypes=[("Classifier pickle file", ".pkl"),
                       ("All files", ".*")]) as fd:
            classifier = cPickle.load(fd)
            input_classes = classifier.get_class_names()
            row = 1 + len(classifier_paths) + sum([len(_) for _ in class_names])
            classifier_paths.append(fd.name)
            label = Tkinter.Label(pane, text=fd.name)
            label.grid(row=row, column=0, columnspan=3, sticky="w")
            row += 1
            d = {}
            for class_name in input_classes:
                label = Tkinter.Label(pane, text=class_name)
                label.grid(row=row, column=1, sticky="e", padx=5)
                entry = Tkinter.Entry(pane)
                entry.insert(0, class_name)
                entry.grid(row=row, column=2, sticky="w")
                d[class_name] = entry
                row += 1
            class_names.append(d)
            bsave.grid(row=row, column=3)
            
    badd = Tkinter.Button(pane, text="Add", command=add_a_classifier)
    badd.grid(row=0, column=1)
            
def go_caffe(pane):
    label = Tkinter.Label(pane, text="Model file")
    label.grid(row=0, column=0)
    model_file_widget = Tkinter.Entry(pane, width=40)
    model_file_widget.grid(row=0, column=1)
    model_file_button = Tkinter.Button(
        pane, text="Select",
        command = lambda: ask_filename(model_file_widget, 
                                       filetypes=[("Model file", ".caffemodel"),
                                                  ("All files", ".*")]))
    model_file_button.grid(row=0, column=2)

    label = Tkinter.Label(pane, text="Prototxt file")
    label.grid(row=1, column=0)
    prototxt_file_widget = Tkinter.Entry(pane, width=40)
    prototxt_file_widget.grid(row=1, column=1)
    prototxt_file_button = Tkinter.Button(
        pane, text="Select",
        command = lambda: ask_filename(
            prototxt_file_widget, 
            filetypes=[("Prototype file", ".prototxt"),
                       ("All files", ".*")]))
    prototxt_file_button.grid(row=1, column=2)

    label = Tkinter.Label(pane, text="Padding")
    label.grid(row=2, column=0)
    padding_widget = XYZWidget(pane, "Padding")
    padding_widget.grid(row=2, column=1, sticky="w")
    
    label = Tkinter.Label(pane, text="Class names")
    label.grid(row=3, column=0)
    class_name_widget = Tkinter.Entry(pane)
    class_name_widget.configure(dict(width=40))
    class_name_widget.grid(row=3, column=1)
    
    label = Tkinter.Label(pane, text="Normalize method")
    label.grid(row=4, column=0)
    normalize_kwds = [_.name for _ in NormalizeMethod]
    normalization_widget = ttk.Combobox(
        pane,
        values = normalize_kwds,
        state="readonly")
    normalization_widget.grid(row=4, column=1)
    normalization_widget.set(NormalizeMethod.NONE.name)
    
    def make_classifier():
        from tkMessageBox import showerror
        model_file = model_file_widget.get()
        if not os.path.exists(model_file):
            showerror("No such model file", "%s does not exist" % model_file)
            return
        prototxt_file = prototxt_file_widget.get()
        if not os.path.exists(prototxt_file):
            showerror("No such weights file", "%s does not exist" % prototxt_file)
        if not padding_widget.validate():
            return
        caffe = all_classifiers["caffe"]
        normalization = NormalizeMethod[normalization_widget.get()]
        class_names = [_.strip() for _ in class_name_widget.get().split(",")]
        classifier = caffe(model_path=model_file,
                           proto_path=prototxt_file,
                           xpad=padding_widget.x,
                           ypad=padding_widget.y,
                           zpad=padding_widget.z,
                           normalize_method=normalization,
                           class_names = class_names)
        return classifier
    
    def save():
        classifier = make_classifier()
        with asksaveasfile(filetypes=[("Pickle file", ".pkl"),
                                      ("All files", ".*")]) as fd:
            cPickle.dump(classifier, fd)
            
    save = Tkinter.Button(pane, text="Save", command=save)
    save.grid(row=5, column=2)
    
if __name__=="__main__":
    main()