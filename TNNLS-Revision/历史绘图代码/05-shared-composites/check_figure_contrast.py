def wcag_contrast(hex1, hex2='#ffffff'):
    def luminance(hex_color):
        hex_color = hex_color.lstrip('#')
        r, g, b = [int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4)]
        def srgb_to_linear(c):
            return c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4
        return 0.2126*srgb_to_linear(r) + 0.7152*srgb_to_linear(g) + 0.0722*srgb_to_linear(b)
    l1, l2 = luminance(hex1), luminance(hex2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)

def grayscale(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return int(0.299*r + 0.587*g + 0.114*b)

print('=== FINAL palette (after update) ===')
colors = {
    'ECG/QTDB': '#0d3b5c',
    'GSC':      '#5d3a8e',
    'pSMNIST':  '#9c4d0e',
}
print('  Color     Hex      Gray  WCAG  Status')
for name, c in colors.items():
    g = grayscale(c)
    cr = wcag_contrast(c)
    pass_label = 'PASS' if cr >= 4.5 else ('PASS-LARGE' if cr >= 3.0 else 'FAIL')
    print(f'  {name:8s} {c}  {g:4d}  {cr:5.2f}  {pass_label}')

print()
print('=== Grayscale distinguishability (task colors) ===')
task_colors = sorted([(n, grayscale(c), c) for n, c in colors.items()], key=lambda x: x[1])
for n, g, c in task_colors:
    print(f'  {n:8s} {c}  gray={g:3d}')
gaps = [task_colors[i+1][1]-task_colors[i][1] for i in range(len(task_colors)-1)]
print(f'Gaps: {gaps}')
print(f'Min gap: {min(gaps)} (target: > 25 for clear grayscale distinction)')

print()
print('=== Panel (c) line distinguishability ===')
# In panel (c), the four lines are: GSC SPRiF (solid+marker), GSC ASRNN (dashed+marker),
# QTDB SPRiF (solid+marker), QTDB ASRNN (dashed+marker)
# Distinguishability sources: color, line style, marker shape
print('GSC color: #5d3a8e (gray=78)  - solid SPRiF, dashed ASRNN, square markers')
print('QTDB color: #0d3b5c (gray=49) - solid SPRiF, dashed ASRNN, circle markers')
print('Grayscale gap: 29 (acceptable)')
print('Additional redundancy: solid/dashed + circle/square markers')
