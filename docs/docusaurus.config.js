module.exports = {
  title: 'BetterLoader',
  tagline: 'The augmented PyTorch dataloader',
  url: 'https://binitai.github.io',
  baseUrl: '/betterloader/',
  onBrokenLinks: 'throw',
  favicon: 'img/favicon.ico',
  organizationName: 'BinItAI',
  projectName: 'BetterLoader',
  themeConfig: {
    navbar: {
      title: 'BetterLoader',
      logo: {
        alt: 'My Site Logo',
        src: 'img/logo.png',
      },
      items: [
        {
          to: 'docs/',
          activeBasePath: 'docs',
          label: 'Docs',
          position: 'left',
        },
        {to: 'blog', label: 'Blog', position: 'left'},
        {
          href: 'https://github.com/binitai/betterloader',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Getting Started',
              to: 'docs/',
            }
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/betterloader',
            },
            {
              label: 'Discord',
              href: 'https://discord.gg/T4Hxcq6',
            }
          ],
        }
      ],
      copyright: `Copyright © ${new Date().getFullYear()} BinIt, Inc. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl:
            'https://github.com/binitai/betterloader',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
