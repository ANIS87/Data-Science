from Tkinter import *
import ttk 
import Tkinter as tk
from PIL import Image, ImageTk
import os, errno
import time
#code
class App:
  def __init__(self, master):
    master.minsize(width=666, height=666)
    master.title("Shopedia data mining")
    master.configure(background='white')
    global frame
    frame = Frame(master,width=500,height=700)
    frame.pack()    
    image= Image.open("Shopedia.jpg")
    image = image.resize((100, 100), Image.ANTIALIAS)
    self.img = ImageTk.PhotoImage(image)         
    label = tk.Label(master, image = self.img)
    label.place(relx=1, rely=0.5)
    label.pack()
    #bar
    self.master=master
    self.image=image
    
    #load data 
    self.button = Button(frame,text="Data", fg="white",command=self.load_data) # command=frame.quit
    self.button.config(bg='black', bd=4, relief=RAISED)
    self.button.pack(side=LEFT)
    
   ###progress bar
    progressbar_rem =  ttk.Progressbar(master,orient=HORIZONTAL,length=200, mode='determinate', value=100)
    progressbar_rem.place(relx=0.35,rely=0.5)
    #progressbar_rem.pack()
    
    self.progressbar_rem=progressbar_rem
    progressbar_vide =  ttk.Progressbar(master,orient=HORIZONTAL,length=200, mode='determinate', value=0)
    progressbar_vide.place(relx=0.35,rely=0.5)
    #progressbar_vide.pack()
    
    self.progressbar_vide=progressbar_vide  
    #Machine Learning
    self.mbutton = Menubutton(frame, text="Machine Learning")     # the pull-down stands alone
    self.picks   = Menu(self.mbutton)               
    self.mbutton.config(menu=self.picks)           
    self.picks.add_command(label='Predictive modeling',  command=self.callback_machine)
    self.picks.add_command(label='Sensitivity analysis',  command=self.callback_machine_sensitivity)    
    self.mbutton.pack()
    self.mbutton.config(bg='green', bd=4, relief=RAISED)
    self.mbutton.pack(side=LEFT)
     ##classifications 
    
    self.mbutton = Menubutton(frame, text="Clustering")     # the pull-down stands alone
    self.picks   = Menu(self.mbutton)               
    self.mbutton.config(menu=self.picks)           
    self.picks.add_command(label='Spatial Clustering',  command=self.callback_classif_two)
    self.picks.add_command(label='Global Clustering',  command=self.callback_classif_All)    
    self.mbutton.pack()
    self.mbutton.config(bg='green', bd=4, relief=RAISED)
    self.mbutton.pack(side=LEFT)
    
    ##statistiques 
    self.mbutton = Menubutton(frame, text="Statistics")     # the pull-down stands alone
    self.picks   = Menu(self.mbutton)               
    self.mbutton.config(menu=self.picks)           
    self.picks.add_command(label='event',  command=self.callback_statsitcs_event)
    self.picks.add_command(label='cpc',  command=self.callback_statsitcs_cpc) 
    self.picks.add_command(label='Time/region',  command=self.callback_statsitcs_Time)    
    #self.mbutton.pack()
    self.mbutton.config(bg='green', bd=4, relief=RAISED)
    self.mbutton.pack(side=LEFT)
    #Help
    self.mbutton = Menubutton(frame, text="Help",fg="white")     # the pull-down stands alone
    self.picks   = Menu(self.mbutton)               
    self.mbutton.config(menu=self.picks) 
    self.picks.add_command(label='Data Collection', command=self.data_informations)
    self.picks.add_command(label='Data Preprocessing', command=self.data_traitement)           
    self.picks.add_command(label='Machine Learning', command=self.data_machine)
    self.picks.add_command(label='Clustering', command=self.data_clustring)
    self.picks.add_command(label='Statistics')     
    self.mbutton.pack()
    self.mbutton.config(bg='blue', bd=4, relief=RAISED)
    self.mbutton.pack(side=LEFT)
    #EXIT
    self.button = Button(frame,text="EXIT", fg="white",command=root.destroy) # command=frame.quit
    self.button.config(bg='red', bd=4, relief=RAISED)
    self.button.pack(side=LEFT)
    self.master=master
    self.image=image
    #self.pack()


  def load_data(self):
    frame=self.master    
    print ("Data loading ...")
    #self.progressbar_vide.start()
    #self.progressbar_vide.pack() 
    #time.sleep(10)   
    execfile("../src/transform_data_csv_2017_CPC.py",locals())
    self.progressbar_vide.destroy()
    self.progressbar_rem=self.progressbar_rem
     
    
    
     

  def callback_machine(self):
    
    frame=self.master    
    print ("Machine Learning ...")
    execfile("../src/Machine_learning_random_forest_System_Pred_AK.py",locals())
     ###figures results 
    if os.path.exists("Machine_learning.png"):
        print("figure existe...")
        image1= Image.open("Machine_learning.png")
    else:
        print("-- trying script")
        
        image1= Image.open("anis.jpeg")
    
    image1 = image1.resize((600, 400), Image.ANTIALIAS)
    self.img1 = ImageTk.PhotoImage(image1)
    self.label1 = tk.Label(frame, image = self.img1)
    #self.slogan.pack(pady=0)
    self.label1.place(relx=0.05, rely=0.3)  
    
  def callback_machine_sensitivity(self):
    frame=self.master
    print ("Machine Learning sensitivity ...")
    execfile("../src/Sensivity_Machine_learning_AK.py",locals())
     ###figures results 
    if os.path.exists("Feature_importances_click.png"):
        print("Figure is shown ......")
        image1= Image.open("Feature_importances_click.png")
    else:
        print("-- trying script")
        
        image1= Image.open("anis.jpeg")
    
    image1 = image1.resize((600, 400), Image.ANTIALIAS)
    self.img1 = ImageTk.PhotoImage(image1)
    self.label1 = tk.Label(frame, image = self.img1)
    #self.slogan.pack(pady=0)
    self.label1.place(relx=0.05, rely=0.3)  
    
  def callback_classif_All(self):
    
    print ("Classification ...")
    execfile("../src/classification_k_prototypes_all_parameter.py",locals())   
    frame=self.master
    
     ###figures results 
    if os.path.exists("classification_k_prototypes_manyfetaures_REFERRER.png"):
        print("Figure is shown ......")
        image1= Image.open("classification_k_prototypes_manyfetaures_REFERRER.png")
    else:
        print("-- trying script")
        
        image1= Image.open("anis.jpeg")
    
    image1 = image1.resize((600, 400), Image.ANTIALIAS)
    self.img1 = ImageTk.PhotoImage(image1)
    self.label1 = tk.Label(frame, image = self.img1)
    #self.slogan.pack(pady=0)
    self.label1.place(relx=0.05, rely=0.3)

  def callback_classif_two(self):
    
    print ("Classification ...")
    execfile("../src/Spataila_K_prototypes_classification.py",locals())   
    frame=self.master
   
     ###figures results 
    if os.path.exists("classification_k_prototypes_Assembla.png"):
        print("Figure is shown ......")
        image1= Image.open("classification_k_prototypes_Assembla.png")
    else:
        print("-- trying script")
        
        image1= Image.open("anis.jpeg")
    
    image1 = image1.resize((600, 400), Image.ANTIALIAS)
    self.img1 = ImageTk.PhotoImage(image1)
    
    self.label1 = tk.Label(frame, image = self.img1)
    #self.slogan.pack(pady=0)
    self.label1.place(relx=0.05, rely=0.3)   

  def callback_statsitcs_event(self):
    
    print ("Statistics event...")
    execfile("../src/statistiques_click_event_by_user.py",locals())   
    frame=self.master
    
     ###figures results 
    if os.path.exists("Statistics_about_visitors.png"):
        print("Figure is shown ......")
        image1= Image.open("Statistics_about_visitors.png")
    else:
        print("-- trying script")
        
        image1= Image.open("anis.jpeg")
    
    image1 = image1.resize((600, 400), Image.ANTIALIAS)
    self.img1 = ImageTk.PhotoImage(image1)
    self.label1 = tk.Label(frame, image = self.img1)
    #self.slogan.pack(pady=0)
    self.label1.place(relx=0.05, rely=0.3)

  def callback_statsitcs_cpc(self):
    
    print ("Statistics cpc...")
    execfile("../src/Statistiques_CPC.py",locals())   
    frame=self.master
    
     ###figures results 
    if os.path.exists("Statistics_cpc_money.png"):
        print("Figure is shown ......")
        image1= Image.open("Statistics_cpc_money.png")
    else:
        print("-- trying script")
        
        image1= Image.open("anis.jpeg")
    
    image1 = image1.resize((600, 400), Image.ANTIALIAS)
    self.img1 = ImageTk.PhotoImage(image1)
    self.label1 = tk.Label(frame, image = self.img1)    
    self.label1.place(relx=0.05, rely=0.3) 


  def callback_statsitcs_Time(self):   
    print ("Statistics time/region...")
    execfile("../src/Diagram_statistics _marque_city_month.py",locals())   
    frame=self.master
    
     ###figures results 
    if os.path.exists("Statistics_time_region_product.png"):
        print("Figure is shown ......")
        image1= Image.open("Statistics_time_region_product.png")
    else:
        print("-- trying script")
        
        image1= Image.open("anis.jpeg")
    
    image1 = image1.resize((600, 400), Image.ANTIALIAS)
    self.img1 = ImageTk.PhotoImage(image1)
    self.label1 = tk.Label(frame, image = self.img1)    
    self.label1.place(relx=0.05, rely=0.3) 






###text to display

  def data_informations(self):
    frame=self.master
    self.data="The raw data are traces of navigation collected from various shopping sites (like www.be.com). The informations collected are: \n \n  -Spatial coordinates of web Visitor ( city, country, longitude and latitude) \n  -Navigator version (mozila, chrome, ..)  \n  -Shopping site (url address) \n  -Temporal parameters (connection date, time spent on shopping site)\n  -Visitor activity (pages seen, click/view)\n  -Visiting benefits (cout par clic)."

    labelx = tk.Label(frame, width=90, height=25,text = self.data,justify=LEFT,  bg = "white", fg="black", font = "Helvetica 10  italic", wraplengt=500)    
    labelx.place(relx=0.05, rely=0.3)



  def data_traitement(self):
    frame=self.master
    self.data="This process aims to prepare data should be used in the machine learning algorithm, such as: \n -The remove of duplicates \n  -Search of empty values (nan value, -, ..) and replace them with average ( case of numeric data : time, longitude, latitude ...) / most probable values ( case of categorical data: city, country) \n  -Normalization of numerical data with respect to the maximum \n  -Transformation of categorical variables into dummy variables. \n more informations: see www.scikit-learn.org/stable/modules/preprocessing.html"


    labely= tk.Label(frame, width=90, height=25,text = self.data, justify=LEFT,  bg = "white", fg="black", font = "Helvetica 10  italic", wraplengt=500)    
    labely.place(relx=0.05, rely=0.3)


  def data_machine(self):
    frame=self.master
    self.data="The  <Machine Learning> button is to create a  predictive model allowing  to predict the action of each web visitor on the current page: it will just see the product or it will make a click. \n This model take as input parameters ( the spatial coordinates, the navigation parameters, the time parameters). \n The algorithms used is the Random Forest tree, which is implemented in the scikit-learn library in python (scikit-learn.org). \n To test this algorithm, the data is divided into two parts : x% of data are used to create the model (20%, 30%, ...)  and the remaining data (80%, 70%,..)  are used to test/validate the model. \n The model fidelity is determined as a function of the accuracy parameter, which is calculated as the number of the true predictions normalized to the total samples (1 means a good model)."


    labely= tk.Label(frame, width=90, height=25,text = self.data, justify=LEFT,  bg = "white", fg="black", font = "Helvetica 10  italic", wraplengt=550)    
    labely.place(relx=0.05, rely=0.3)

  def data_clustring(self):
    frame=self.master
    self.data="The clustering algorithm used in this step is the K-prototypes algorithm, which is very similar to the popular k-means algorithm. It's major advantage is that it allows to classify a mixed data (categorical + numeric features). This algorithm is available into python version  ( see the Kmodes project given in  www.github.com/nicodv/kmodes)."


    labely= tk.Label(frame, width=90, height=25,text = self.data, justify=LEFT,  bg = "white", fg="black", font = "Helvetica 10  italic", wraplengt=550)    
    labely.place(relx=0.0, rely=0.3)


###Principal code 
##delete File is just option 
#def silentremove(filename):
#    try:
#        os.remove(filename)
#    except OSError as e: # this would be "except OSError, e:" before Python 2.6
#        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
#            raise # re-raise exception if a different error occured

#silentremove('data_brut.csv')
root = Tk()
app = App(root)
root.mainloop()
