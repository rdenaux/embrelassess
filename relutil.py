def calc_rel_type_for_dfrow_fn(rel_name_field='name', kg_field='KG'):
    """Returns a function that calculates the rel_type for a df_row
    The row must have a field for the rel_name 
    """
    def _calc_rel_type_for(df_row):
        kg_name = df_row[kg_field]
        rel_name = df_row[rel_name_field]
        if kg_name == 'sensi':
            if rel_name in ['rel', 'syn-sensiDomain']:
                return 'categ'
            elif rel_name in ['supernomen-subnomen', 'superverbum-subverbum']:
                return 'hypernym'
            elif rel_name in ['noun-antonym', 'verb-antonym', 'adjective-antonym']: 
                return 'similarity'
            elif rel_name in ['syncon-cause', 'syncon-implication', 'syncon-corpus', 'syncon-unification']:
                return 'similarity'
            elif rel_name in ['synonym', 'lemma']:
                return 'synonymy'
            elif rel_name in ['omninomen-parsnomen', 'omninomen-parsnomen_(m)']:
                return 'meronymy'
            elif rel_name in ['verb-object', 'verb-subject']:
                return 'positional'
            elif rel_name in ['Verb', 'Noun', 'ProperNoun', 'Adverb', 'Adjective']:
                return 'part-of-speech'
            elif rel_name in ['noun+against-noun',
                       'noun+as-noun',
                       'noun+at-noun', 
                       'noun+between-noun', 
                       'noun+by-noun', 
                       'noun+for-noun', 
                       'noun+from-noun', 
                       'noun+in-noun', 
                       'noun+of-noun', 
                       'noun+on-above-noun', 
                       'noun+to-noun', 
                       'noun+with-noun']:
                return 'prepositional'
            elif rel_name in ['verb+against-noun', 
                       'verb+at-noun', 
                       'verb+for-noun', 
                       'verb+from-noun', 
                       'verb+in-noun', 
                       'verb+of-noun', 
                       'verb+on-noun', 
                       'verb+to-noun', 
                       'verb+with-noun', 
                       'verb-noun-person']:
                return 'prepositional'
            elif rel_name in ['verb_(prep)-verb']:
                return 'prepositional'
            elif rel_name in ['adjective+for-noun',
                                    'adjective+to-noun',
                       'adjective+in-noun',
                       'adjective+of-noun']:
                return 'prepositional'
            elif rel_name in ['adverb-noun', 'adverb-verb', 'adverb-adjective', 'adverb-adverb']:
                return 'positional'
            elif rel_name in ['adjective-class']:
                return 'positional'
            elif rel_name in ['syncon-geography']:
                return 'geography'
            elif rel_name.startswith('random_'):
                return 'random'
            else:
                print('Could not assign a type to sensi rel with name', rel_name)
                return 'sensiRelType'
        elif kg_name == 'wnet':
            if rel_name in ['category_domain', 'region_domain' , 'usage_domain',
                                 'member_of_category_domain', 'member_of_region_domain', 'member_of_usage_domain']: #inverse
                return 'categ'
            elif rel_name in ['hypernym', 'instance_hypernym',
                                   'hyponym', 'instances_hyponym']: #inverse rels
                return 'hypernym'
            elif rel_name in ['also_see', 'antonym', 'derivation', 'pertainym', 'participle_of']:
                return 'lexicalRel'
            elif rel_name in ['attribute', 'similar', 'cause', 'entailment', 'verb_group']:
                return 'similarity'
            elif rel_name in ['synonym']:
                return 'synonymy'
            elif rel_name in ['member_meronym',  'substance_meronym' , 'part_meronym',
                                   'member_holonym', 'substance_holonym', 'part_holonym']: #inverse
                return 'meronymy'
            elif rel_name.startswith('random_'):
                return 'random'
            else:
                return 'wnetRelType'
        else:
            raise Exception('unsupported KG name value', kg_name)
    return _calc_rel_type_for
