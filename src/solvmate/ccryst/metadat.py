import xmltodict
import hashlib

from dataclasses import dataclass
from rdkit import Chem


@dataclass
class Reaction:
    inchi_product: str
    smiles_product: str
    text_recipe: str

    def to_dict(self) -> dict:
        return {
            "inchi_product": self.inchi_product,
            "smiles_product": self.smiles_product,
            "text_recipe": self.text_recipe,
        }


def parse_reaction_xml(xml: str) -> list[Reaction]:
    """
    The following example shows what a typical lowe reaction
    xml entry looks like:
    >>> xml = _test_reaction_xml()
    >>> xmltodict.parse(xml) # doctest: +NORMALIZE_WHITESPACE
    {'reactionList':
    {'reaction':
        {'dl:source':
            {'dl:documentId': 'USRE042477E1',
            'dl:headingText': 'Methyl 2-hydroxy-3-methoxy-3,3-diphenylpropionate',
            'dl:paragraphNum': '0152',
            'dl:paragraphText': '5 g (19.6 mmol) of methyl 3,3-diphenyl-2,3-epoxypropionate were dissolved in 50 ml of absolute methanol and, at 0° C., 0.1 ml of boron trifluoride etherate was added. The mixture was stirred at 0° C. for 2 h and at room temperature for a further 12 h. The solvent was distilled out, the residue was taken up in ethyl acetate, washed with sodium bicarbonate solution and water and dried over magnesium sulfate. After removal of the solvent by distillation there remained 5.5 g (88%) of a pale yellow oil.'
            },
            'dl:reactionSmiles': '[C:1]1([C:7]2([C:14]3[CH:19]=[CH:18][CH:17]=[CH:16][CH:15]=3)[O:13][CH:8]2[C:9]([O:11][CH3:12])=[O:10])[CH:6]=[CH:5][CH:4]=[CH:3][CH:2]=1.B(F)(F)F.C[CH2:25][O:26]CC>CO>[OH:13][CH:8]([C:7]([O:26][CH3:25])([C:1]1[CH:2]=[CH:3][CH:4]=[CH:5][CH:6]=1)[C:14]1[CH:19]=[CH:18][CH:17]=[CH:16][CH:15]=1)[C:9]([O:11][CH3:12])=[O:10] |f:1.2|',
            'productList':
                {'product':
                    {'molecule': {'@id': 'm0', 'name': {'@dictRef': 'nameDict:unknown',
                    '#text': 'Methyl 2-hydroxy-3-methoxy-3,3-diphenylpropionate'}},
                    'identifier': [
                        {'@dictRef': 'cml:smiles', '@value': 'OC(C(=O)OC)C(C1=CC=CC=C1)(C1=CC=CC=C1)OC'},
                        {'@dictRef': 'cml:inchi', '@value': 'InChI=1S/C17H18O4/c1-20-16(19)15(18)17(21-2,13-9-5-3-6-10-13)14-11-7-4-8-12-14/h3-12,15,18H,1-2H3'}
                        ]}
                    },
                'reactantList': {
                            'reactant': [{'@role': 'reactant', '@count': '1', 'molecule': {'@id': 'm1', 'name': {'@dictRef': 'nameDict:unknown', '#text': 'methyl 3,3-diphenyl-2,3-epoxypropionate'}}, 'amount': [{'@dl:propertyType': 'AMOUNT', '@dl:normalizedValue': '0.0196', '#text': '19.6 mmol'}, {'@dl:propertyType': 'MASS', '@dl:normalizedValue': '5', '#text': '5 g'}], 'dl:entityType': 'exact',
                            'identifier': [{'@dictRef': 'cml:smiles', '@value': 'C1(=CC=CC=C1)C1(C(C(=O)OC)O1)C1=CC=CC=C1'}, {'@dictRef': 'cml:inchi', '@value': 'InChI=1S/C16H14O3/c1-18-15(17)14-16(19-14,12-8-4-2-5-9-12)13-10-6-3-7-11-13/h2-11,14H,1H3'}]}, {'@role': 'reactant', '@count': '1', 'molecule': {'@id': 'm2', 'name': {'@dictRef': 'nameDict:unknown', '#text': 'boron trifluoride etherate'}}, 'amount': {'@dl:propertyType': 'VOLUME', '@dl:normalizedValue': '0.0001', '#text': '0.1 ml'},
                            'identifier': [{'@dictRef': 'cml:smiles', '@value': 'B(F)(F)F.CCOCC'}, {'@dictRef': 'cml:inchi', '@value': 'InChI=1S/C4H10O.BF3/c1-3-5-4-2;2-1(3)4/h3-4H2,1-2H3;'}], 'dl:entityType': 'exact'}]},
                'spectatorList': {'spectator': {'@role': 'solvent', 'molecule': {'@id': 'm3', 'name': {'@dictRef': 'nameDict:unknown', '#text': 'methanol'}}, 'amount': {'@dl:propertyType': 'VOLUME', '@dl:normalizedValue': '0.050', '#text': '50 ml'}, 'identifier': [{'@dictRef': 'cml:smiles', '@value': 'CO'}, {'@dictRef': 'cml:inchi', '@value': 'InChI=1S/CH4O/c1-2/h2H,1H3'}], 'dl:entityType': 'exact'}}}}}

    We can see that it is indeed quite complex. But for now,
    we are just interested in the inchi of the product and the
    text describing the reaction. Hence, it should be quite
    simple to extract the desired information from above xml.

    """
    skip_count = 0
    xml = xmltodict.parse(xml)
    if "reaction" not in xml["reactionList"]:
        return
    if isinstance(xml["reactionList"]["reaction"], dict):
        # in this case there was only a single reaction...
        xml["reactionList"]["reaction"] = [xml["reactionList"]["reaction"]]
    for reaction in xml["reactionList"]["reaction"]:
        source = reaction["dl:source"]
        text = source["dl:paragraphText"]
        product = reaction["productList"]["product"]
        if isinstance(product, list):
            skip_count += 1
            continue

        product_ids = product["identifier"]
        product_inchi = None
        product_smiles = None
        for pid in product_ids:
            if pid["@dictRef"] == "cml:inchi":
                product_inchi = pid["@value"]
            if pid["@dictRef"] == "cml:smiles":
                product_smiles = pid["@value"]
        if not product_inchi and not product_smiles:
            continue
        if product_smiles and not product_inchi:
            product_inchi = Chem.MolToInchi(Chem.MolFromSmiles(product_smiles))
        if product_inchi and not product_smiles:
            product_smiles = Chem.MolToSmiles(Chem.MolFromInchi(product_inchi))

        yield Reaction(
            inchi_product=product_inchi,
            smiles_product=product_smiles,
            text_recipe=text,
        )

    print("skipped", skip_count, "/", len(xml["reactionList"]["reaction"]), "reactions")


def _test_reaction_xml() -> str:
    return """
  <reactionList>
  <reaction>
    <dl:source>
      <dl:documentId>USRE042477E1</dl:documentId>
      <dl:headingText>Methyl 2-hydroxy-3-methoxy-3,3-diphenylpropionate</dl:headingText>
      <dl:paragraphNum>0152</dl:paragraphNum>
      <dl:paragraphText>5 g (19.6 mmol) of methyl 3,3-diphenyl-2,3-epoxypropionate were dissolved in 50 ml of absolute methanol and, at 0° C., 0.1 ml of boron trifluoride etherate was added. The mixture was stirred at 0° C. for 2 h and at room temperature for a further 12 h. The solvent was distilled out, the residue was taken up in ethyl acetate, washed with sodium bicarbonate solution and water and dried over magnesium sulfate. After removal of the solvent by distillation there remained 5.5 g (88%) of a pale yellow oil.</dl:paragraphText>
    </dl:source>
    <dl:reactionSmiles>[C:1]1([C:7]2([C:14]3[CH:19]=[CH:18][CH:17]=[CH:16][CH:15]=3)[O:13][CH:8]2[C:9]([O:11][CH3:12])=[O:10])[CH:6]=[CH:5][CH:4]=[CH:3][CH:2]=1.B(F)(F)F.C[CH2:25][O:26]CC&gt;CO&gt;[OH:13][CH:8]([C:7]([O:26][CH3:25])([C:1]1[CH:2]=[CH:3][CH:4]=[CH:5][CH:6]=1)[C:14]1[CH:19]=[CH:18][CH:17]=[CH:16][CH:15]=1)[C:9]([O:11][CH3:12])=[O:10] |f:1.2|</dl:reactionSmiles>
    <productList>
      <product>
        <molecule id="m0">
          <name dictRef="nameDict:unknown">Methyl 2-hydroxy-3-methoxy-3,3-diphenylpropionate</name>
        </molecule>
        <identifier dictRef="cml:smiles" value="OC(C(=O)OC)C(C1=CC=CC=C1)(C1=CC=CC=C1)OC"/>
        <identifier dictRef="cml:inchi" value="InChI=1S/C17H18O4/c1-20-16(19)15(18)17(21-2,13-9-5-3-6-10-13)14-11-7-4-8-12-14/h3-12,15,18H,1-2H3"/>
      </product>
    </productList>
    <reactantList>
      <reactant role="reactant" count="1">
        <molecule id="m1">
          <name dictRef="nameDict:unknown">methyl 3,3-diphenyl-2,3-epoxypropionate</name>
        </molecule>
        <amount dl:propertyType="AMOUNT" dl:normalizedValue="0.0196">19.6 mmol</amount>
        <amount dl:propertyType="MASS" dl:normalizedValue="5">5 g</amount>
        <dl:entityType>exact</dl:entityType>
        <identifier dictRef="cml:smiles" value="C1(=CC=CC=C1)C1(C(C(=O)OC)O1)C1=CC=CC=C1"/>
        <identifier dictRef="cml:inchi" value="InChI=1S/C16H14O3/c1-18-15(17)14-16(19-14,12-8-4-2-5-9-12)13-10-6-3-7-11-13/h2-11,14H,1H3"/>
      </reactant>
      <reactant role="reactant" count="1">
        <molecule id="m2">
          <name dictRef="nameDict:unknown">boron trifluoride etherate</name>
        </molecule>
        <amount dl:propertyType="VOLUME" dl:normalizedValue="0.0001">0.1 ml</amount>
        <identifier dictRef="cml:smiles" value="B(F)(F)F.CCOCC"/>
        <identifier dictRef="cml:inchi" value="InChI=1S/C4H10O.BF3/c1-3-5-4-2;2-1(3)4/h3-4H2,1-2H3;"/>
        <dl:entityType>exact</dl:entityType>
      </reactant>
    </reactantList>
    <spectatorList>
      <spectator role="solvent">
        <molecule id="m3">
          <name dictRef="nameDict:unknown">methanol</name>
        </molecule>
        <amount dl:propertyType="VOLUME" dl:normalizedValue="0.050">50 ml</amount>
        <identifier dictRef="cml:smiles" value="CO"/>
        <identifier dictRef="cml:inchi" value="InChI=1S/CH4O/c1-2/h2H,1H3"/>
        <dl:entityType>exact</dl:entityType>
      </spectator>
    </spectatorList>
</reaction>
</reactionList>
"""
