using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Web;

namespace KourseWork.Models
{
    public class UserContext : DbContext
    {
        public UserContext() :
            base("MyConnection1")
        { }
        public DbSet<User> Users { get; set; }
    }
}