from biopandas.pdb import PandasPdb
from pathlib import Path

class pdb_analysis(object):
    def __init__(self,pdb_file_code=False,pdb_path=False, output_path=False):
        self.pdb_file_code = pdb_file_code
        self.pdb_path = pdb_path
        self.output_path = output_path

    # self.load_pdb()

    def load_pdb(self):
        ppdb = PandasPdb()

        if self.pdb_file_code is False:
            print('Error -- please provide a pdb file')
            exit()
        elif self.pdb_path is False:
            print('pdb file fetched from the database')
            try:
                pdb_file = ppdb.fetch_pdb(self.pdb_file_code)
            except:
                print('Error -- please check the pdb code provided')
                exit()
        else: 
            try:
                filename = Path(self.pdb_path,self.pdb_file_code+'.pdb')
                pdb_file = ppdb.read_pdb(str(filename))
            except: 
                print('Error -- please check the pdb code and/or path')
                exit()
        return pdb_file

    def mutation(self, ref_residue=False, reslist=False):

        backbone_atoms = ['N','CA','C','O']
        if reslist is False:
            self.pdb_file.df['ATOM'] = self.pdb_file.df['ATOM'][self.pdb_file.df['ATOM']['atom_name'].isin(backbone_atoms)]
            self.pdb_file.df['ATOM'].loc[:, 'residue_name'] = 'GLY'

            if self.output_path is False:
                filename_output = Path('~/',self.pdb_file_code+'_Gly.pdb')
            else:
                filename_output = Path(self.output_path,self.pdb_file_code+'_Gly.pdb')

        else:
            for i in reslist:
                pdb_file = self.load_pdb()

                residue_no_mutation = [ref_residue,i]
                df_no_mutation = pdb_file.df['ATOM'][pdb_file.df['ATOM']['residue_number'].isin(residue_no_mutation)]

                pdb_file.df['ATOM'] = pdb_file.df['ATOM'][(pdb_file.df['ATOM']['residue_number'].isin(residue_no_mutation)==False) & (pdb_file.df['ATOM']['atom_name'].isin(backbone_atoms))]

                pdb_file.df['ATOM'].loc[:, 'residue_name'] = 'GLY'

                pdb_file.df['ATOM'] = pdb_file.df['ATOM'].merge(df_no_mutation, how='outer')
                pdb_file.df['ATOM'] = pdb_file.df['ATOM'].sort_values('residue_number')

                if self.output_path is False:
                    filename_output = Path('~/',self.pdb_file_code+'_Gly.pdb')
                else:
                    filename_output = Path(self.output_path,self.pdb_file_code+'_Gly_'+str(ref_residue)+'_'+str(i)+'.pdb')


                pdb_file.to_pdb(path=filename_output, 
                    records=['ATOM'], 
                    gz=False, 
                    append_newline=True)


