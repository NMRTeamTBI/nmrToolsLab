
class UtilsHandler():
    def __init__(self):
        pass

    def cm2inch(self,*tupl):
        inch = 2.54
        if isinstance(tupl[0], tuple):
            return tuple(i/inch for i in tupl[0])
        else:
            return tuple(i/inch for i in tupl)
   
    def three_to_one(self):
        
        three_to_one = {
            'CYS': 'C',
            'ASP': 'D',
            'SEP': 'S',
            'SER': 'S',
            'GLN': 'Q',
            'LYS': 'K',
            'PRO': 'P',
            'THR': 'T',
            'PHE': 'F',
            'ALA': 'A',
            'HIS': 'H',
            'GLY': 'G',
            'ILE': 'I',
            'GLU': 'E',
            'LEU': 'L',
            'ARG': 'R',
            'TRP': 'W',
            'VAL': 'V',
            'ASN': 'N',
            'TYR': 'Y',
            'MET': 'M'
        }

        return three_to_one
    
    def one_to_three(self):
        
        one_to_three = {
            'C':'CYS',
            'D':'ASP',
            'S':'SEP',
            'S':'SER',
            'Q':'GLN',
            'K':'LYS',
            'P':'PRO',
            'T':'THR',
            'F':'PHE',
            'A':'ALA',
            'H':'HIS',
            'G':'GLY',
            'I':'ILE',
            'E':'GLU',
            'L':'LEU',
            'R':'ARG',
            'W':'TRP',
            'V':'VAL',
            'N':'ASN',
            'Y':'TYR',
            'M':'MET'
        }

        return one_to_three