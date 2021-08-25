import os
import glob
import shutil 
from os.path import expanduser
from pathlib import Path




def copy_fichier():
    list_dirs = sorted(glob.glob(os.path.join('./', "checkpoint*")))
    
    home = str(Path.home())
    
    where_to_save = home +  '/checkpoint_save'
        
    if not os.path.exists(where_to_save):
        os.makedirs(where_to_save)
    
    for rep in list_dirs:
        print(rep)
        list_txt_files = sorted(glob.glob(os.path.join(rep, "*.txt")))
        name_rep = rep.split('/')
        name_rep = name_rep[len(name_rep) - 1]
        
        new_where_to_save = where_to_save + '/' + name_rep
        
        print(new_where_to_save)
        
        if not os.path.exists(new_where_to_save):
            os.makedirs(new_where_to_save)
            
        i = 0
        for txt in list_txt_files:
            dest_name = "logs_train.txt" if (i == 0) else "logs_val.txt"
            dest_name = new_where_to_save +  '/' + dest_name
            print(shutil.copyfile(txt, dest_name) )
            
            i = i + 1
        
        print('\n')
        
        

    
    
if __name__=='__main__':
    copy_fichier()