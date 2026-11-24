content = open(r'd:\DOP\sapec_paper.tex', encoding='utf-8').read()

content = content.replace('0.8553', '0.8424')
content = content.replace('106.59', '111.26')
content = content.replace('+20.44', '+16.95')
content = content.replace('20.44\\%', '16.95\\%')
content = content.replace('x^{*}=60~min, t_d^{*}=5~min', 'x^{*}=45~min, t_d^{*}=5~min')
content = content.replace('(x^{*}=60', '(x^{*}=45')
content = content.replace('x^*=60', 'x^*=45')

open(r'd:\DOP\sapec_paper.tex', 'w', encoding='utf-8').write(content)

import re
hits = [m.group() for m in re.finditer(r'0\.855[0-9]|106\.5[0-9]|20\.44', content)]
print('Remaining old hits:', len(hits), hits)
print('x*=45:', 'x^{*}=45' in content)
print('0.8424:', '0.8424' in content)
print('111.26:', '111.26' in content)
print('16.95:', '16.95' in content)
