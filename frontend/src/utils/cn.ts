type ClassValue =
  | string
  | number
  | bigint
  | boolean
  | null
  | undefined
  | { [key: string]: boolean | undefined | null };

export function cn(...inputs: ClassValue[]): string {
  const classes: string[] = [];
  for (const input of inputs) {
    if (typeof input === 'boolean') continue;
    if (typeof input === 'string' && input.trim()) {
      classes.push(input.trim());
    } else if (typeof input === 'object' && input !== null) {
      for (const [key, value] of Object.entries(input)) {
        if (value) classes.push(key);
      }
    }
  }
  return classes.join(' ');
}
